import json
import os
import numpy as np
import yaml
import TinyLensGpu.Profile.util as util


# Parameter order of the sigma array (corresponds to the frontend p vector)
SIGMA_ORDER = [
    "lens_x", "lens_y", "theta_e", "lens_ell", "lens_ang",
    "source_x", "source_y", "source_size", "source_ell", "source_ang",
    "n_sersic", "Ie",
]


def convert_to_array(frontData):
    if not isinstance(frontData, dict) or 'components' not in frontData:
        print("Invalid input data structure: missing the components field")
        return
    if not isinstance(frontData['components'], list):
        print("components should be a list")
        return

    lens_params = next((c for c in frontData['components'] if isinstance(c, dict) and c.get('plane') == 'lens'), None)
    source_params = next((c for c in frontData['components'] if isinstance(c, dict) and c.get('plane') == 'source'), None)
    if not isinstance(lens_params, dict) or not isinstance(source_params, dict):
        print("Unable to find lens or source parameters in a valid format")
        return

    pixscale = frontData.get("pixscale", None)
    pixscale = float(pixscale) if pixscale is not None else None

    update_model_config(lens_params, source_params, pixscale)


def sanitize_data(data):
    return json.loads(json.dumps(data))


def safe_round(value, default=0.0, decimals=4):
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return round(default, decimals)


def parse_sigma_array(sigma_array):
    """Unpack the sigma array into a dict following SIGMA_ORDER; return None if the array is missing or has the wrong length"""
    if sigma_array is None or len(sigma_array) != len(SIGMA_ORDER):
        return None
    return {name: float(sigma_array[i]) for i, name in enumerate(SIGMA_ORDER)}


def propagate_to_e1e2(ang_deg, ell):
    """
    Best-fit mapping from (ang, ell) to (e1, e2).
    ell = 1/q; the frontend convention is ell > 1, so it must be converted to q internally before the transformation.
    Returns (e1, e2) without the sign flip (handled by the caller).
    """
    e1_raw, e2_raw = util.phi_q2_ellipticity(ang_deg * np.pi / 180, ell)
    return float(e1_raw), float(e2_raw)


def make_uniform(center, half_width, hard_limits):
    low = max(center - half_width, hard_limits[0])
    high = min(center + half_width, hard_limits[1])
    return {
        "prior_type": "uniform",
        "prior_settings": [safe_round(low), safe_round(high)],
        "limits": list(hard_limits),
        "fixed": False,
    }


def make_gaussian(center, sigma, hard_limits):
    return {
        "prior_type": "gaussian",
        "prior_settings": [safe_round(center), safe_round(sigma)],
        "limits": list(hard_limits),
        "fixed": False,
    }


def update_model_config(lens_params, source_params, pixscale,
                        config_file="model_config.yaml"):
    """Update model_config.yaml with the Hoopla best-fit (cao double-Sersic configuration)"""
    try:
        if not os.path.exists(config_file):
            print(f"Config file {config_file} does not exist.")
            return
        if not isinstance(lens_params, dict) or not isinstance(source_params, dict):
            print("lens_params / source_params must be dicts")
            return

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # ========== LENS MASS (SIE) ==========
        lens_mass_list = config["model_components"]["lens_mass_list"]
        if lens_mass_list:
            theta_e = safe_round(lens_params.get("theta_e", 0.0))
            ell = safe_round(lens_params.get("ell", 1.0))
            ang = safe_round(lens_params.get("ang", 0.0))

            # theta_E: uniform, ±30%
            w_te = abs(theta_e) * 0.3
            lens_mass_list[0]["params"]["theta_E"] = make_uniform(theta_e, w_te, [0.0, 10.0])

            # e1, e2: gaussian, sigma=0.3
            e1, e2 = propagate_to_e1e2(ang, ell)
            e1, e2 = -e1, -e2  # sign flip required by the frontend ell>1 convention
            lens_mass_list[0]["params"]["e1"] = make_gaussian(e1, 0.3, [-1.0, 1.0])
            lens_mass_list[0]["params"]["e2"] = make_gaussian(e2, 0.3, [-1.0, 1.0])

            # lens center: gaussian, sigma=0.1
            x_l = safe_round(lens_params.get("x", 0.0))
            y_l = safe_round(lens_params.get("y", 0.0))
            lens_mass_list[0]["params"]["center_x"] = make_gaussian(x_l, 0.1, [-1.0, 1.0])
            lens_mass_list[0]["params"]["center_y"] = make_gaussian(y_l, 0.1, [-1.0, 1.0])

        # ========== SOURCE LIGHT ==========
        # First Sersic (n=1): receives the frontend source parameters
        # Second Sersic (n=4): e1/e2/center are linked to the first one and inherited automatically
        source_light_list = config["model_components"]["source_light_list"]
        if source_light_list:
            Re = safe_round(source_params.get("size", 0.0))
            # n_sersic is passed in from the frontend but not processed (already fixed in the config)
            ell_s = safe_round(source_params.get("ell", 1.0))
            ang_s = safe_round(source_params.get("ang", 0.0))
            x_s = safe_round(source_params.get("x", 0.0))
            y_s = safe_round(source_params.get("y", 0.0))

            # R_sersic: uniform, ±30%
            w_re = abs(Re) * 0.3
            source_light_list[0]["params"]["R_sersic"] = make_uniform(Re, w_re, [0.0, 10.0])

            # e1, e2: gaussian, sigma=0.3
            e1s, e2s = propagate_to_e1e2(ang_s, ell_s)
            e1s, e2s = -e1s, -e2s
            source_light_list[0]["params"]["e1"] = make_gaussian(e1s, 0.3, [-1.0, 1.0])
            source_light_list[0]["params"]["e2"] = make_gaussian(e2s, 0.3, [-1.0, 1.0])

            # source center: gaussian, sigma=0.5
            source_light_list[0]["params"]["center_x"] = make_gaussian(x_s, 0.5, [-3.5, 3.5])
            source_light_list[0]["params"]["center_y"] = make_gaussian(y_s, 0.5, [-3.5, 3.5])

        # ========== pixel_scale ==========
        if pixscale is not None:
            config["dataset"]["pixel_scale"] = safe_round(pixscale)

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sanitize_data(config), f,
                      allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"Config file {config_file} updated successfully.")
    except Exception as e:
        print(f"Error occurred while updating the config file: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    frontData = load_json_file()
    if frontData:
        convert_to_array(frontData)
