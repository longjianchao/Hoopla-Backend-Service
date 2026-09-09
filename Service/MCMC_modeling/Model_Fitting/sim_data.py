
import sys
import os
import json
import numpy as np
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from TinyLensGpu.Simulator.Image import Simulator
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
from TinyLensGpu.Profile.Mass.Sie import SIE
from TinyLensGpu.Profile.Mass.Shear import Shear
from TinyLensGpu.Simulator.Image.Simulator import SimulatorConfig
from TinyLensGpu.Simulator.Image.Simulator import util as su
from TinyLensGpu.Profile.Light.Gaussian import Gaussian
import TinyLensGpu.Profile.util as util
from astropy.io import fits

# Set the random seed for reproducibility
np.random.seed(42)

# Create the physical model
phy_model = Simulator.PhysicalModel(
    lens_mass=[SIE(), Shear()],
    source_light=[SersicEllipse()],
    lens_light=[SersicEllipse()],
)

# Generate the PSF kernel
x_psf, y_psf = su.make_grid_2d(21, 0.05)
psf_kernel = Gaussian().light(x_psf, y_psf, 1.0, 0.05, 0.0, 0.0)
psf_kernel /= psf_kernel.sum()
psf_kernel = np.asarray(psf_kernel)
# psf_kernel = np.array([ [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

# Configure the simulator
sim_config = SimulatorConfig(
    dpix=0.05,
    npix=200,
    psf_kernel=psf_kernel,
    nsub=1,
)

# Create the simulator object
sim_obj = Simulator.LensSimulator(
    phy_model,
    sim_config
)

def mock_lens(ideal_image, back_rms, exp_time):
    """Add noise to the ideal image"""
    noise_map = np.sqrt(ideal_image/exp_time + back_rms**2)
    noisy_image = ideal_image + np.random.normal(0, noise_map)
    return noisy_image, noise_map

def generate_random_parameters():
    """Generate random simulation parameters with a uniform parameter distribution"""
    # Randomly generate the lens mass parameters
    phi_l = np.random.uniform(0, 2*np.pi)    # position angle
    q_l = np.random.uniform(0.3, 1.0)        # axis ratio
    e1_l, e2_l = util.phi_q2_ellipticity(phi_l, q_l)
    
    lens_mass_par_list = [
        dict(
            theta_E=np.random.uniform(0.5, 3.0),
            e1=e1_l,
            e2=e2_l,
            center_x=0.0,  # fixed at the origin
            center_y=0.0,  # fixed at the origin
        ),
        dict(
            gamma1=0.0,
            gamma2=0.0,
        ),  
    ]
    phi_s = np.random.uniform(0, 2*np.pi)    # position angle
    q_s = np.random.uniform(0.3, 1.0)        # axis ratio
    e1_s, e2_s = util.phi_q2_ellipticity(phi_s, q_s)
    
    # Randomly generate the source light parameters
    source_light_par_list = [
        dict(
            R_sersic=np.random.uniform(0.05, 0.8),
            n_sersic=np.random.uniform(0.3, 2.5),
            e1=e1_s,
            e2=e2_s,
            center_x=np.random.uniform(-0.5, 0.5),
            center_y=np.random.uniform(-0.5, 0.5),
            Ie=1,
        )
        # dict(
        #     R_sersic=np.random.uniform(0.05, 0.8),
        #     n_sersic=np.random.uniform(0.5, 4.0),
        #     e1=0.0,
        #     e2=0.0,
        #     center_x=0,
        #     center_y=0,
        #     Ie=1,
        # )
    ]
    
    # # Randomly generate the lens light parameters
    # lens_light_par_list = [
    #     dict(
    #         R_sersic=np.random.uniform(0.5, 2.0),
    #         n_sersic=np.random.uniform(2.0, 6.0),
    #         e1=np.random.uniform(-0.3, 0.3),
    #         e2=np.random.uniform(-0.3, 0.3),
    #         center_x=0.0,  # fixed at the origin
    #         center_y=0.0,  # fixed at the origin
    #         Ie=np.random.uniform(0.5, 2.0),  
    #     )
    # ]
    lens_light_par_list = []
    
    # Randomly generate the noise parameters
    noise_params = {
        'back_rms': 0.05,
        'exp_time': 300
    }
    
    params = [
        lens_mass_par_list,
        source_light_par_list,
        lens_light_par_list,
    ]
    
    # Consolidate all parameters into one dictionary for saving
    all_params = {
        'lens_mass': lens_mass_par_list,
        'source_light': source_light_par_list,
        'lens_light': lens_light_par_list,
        'noise_params': noise_params
    }
    
    return params, all_params

def main():
    # Get the directory containing the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the sim_data top-level folder
    sim_data_dir = os.path.join(current_dir, 'sim_data')
    os.makedirs(sim_data_dir, exist_ok=True)
    
    # Generate 100 sets of simulation data
    for i in range(1, 101):
        # Create the index folder
        folder_name = str(i)
        folder_path = os.path.join(sim_data_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"Generating simulation data set {i}...")
        
        # Generate random parameters
        params, all_params = generate_random_parameters()
        
        # Run the simulation
        img_2d = sim_obj.simulate(
            params,
            xgrid_sub=sim_obj.xgrid_sub,
            ygrid_sub=sim_obj.ygrid_sub,
            psf_kernel=psf_kernel
        )
        
        # Add noise
        back_rms = all_params['noise_params']['back_rms']
        exp_time = all_params['noise_params']['exp_time']
        noisy_image, noise_map = mock_lens(img_2d, back_rms, exp_time)
        
        # # Compute and print the SNR
        # signal = np.sum(img_2d)
        # noise = np.sqrt(np.sum(noise_map ** 2))
        # snr = signal / noise if noise != 0 else 0
        # print(f"  SNR: {snr:.2f}")
        
        # Save the data to fits files
        fits.writeto(os.path.join(folder_path, 'image.fits'), np.asarray(noisy_image), overwrite=True)
        fits.writeto(os.path.join(folder_path, 'noise.fits'), np.asarray(noise_map), overwrite=True)
        fits.writeto(os.path.join(folder_path, 'psf.fits'), np.asarray(psf_kernel), overwrite=True)
        
        # Save the parameters to a json file
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            # Handle numpy integers and floats
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            # Handle numpy arrays and ArrayImpl types
            elif hasattr(obj, 'shape'):  # check whether it is an array type
                return obj.tolist()
            # Handle dictionaries
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            # Handle lists
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            # Return other types as-is
            else:
                return obj
        
        # Build a new dictionary containing only the parameter information to be saved, avoiding serialization issues
        params_to_save = {
            'lens_mass': [
                {
                    'theta_E': float(all_params['lens_mass'][0]['theta_E']),
                    'e1': float(all_params['lens_mass'][0]['e1']),
                    'e2': float(all_params['lens_mass'][0]['e2']),
                    'center_x': 0.0,
                    'center_y': 0.0
                },
                {
                    'gamma1': float(all_params['lens_mass'][1]['gamma1']),
                    'gamma2': float(all_params['lens_mass'][1]['gamma2'])
                }
            ],
            'source_light': [
                {
                    'R_sersic': float(all_params['source_light'][0]['R_sersic']),
                    'n_sersic': float(all_params['source_light'][0]['n_sersic']),
                    'e1': float(all_params['source_light'][0]['e1']),
                    'e2': float(all_params['source_light'][0]['e2']),
                    'center_x': float(all_params['source_light'][0]['center_x']),
                    'center_y': float(all_params['source_light'][0]['center_y']),
                    'Ie': float(all_params['source_light'][0]['Ie'])
                }
            ],
            # 'lens_light': [
            #     {
            #         'R_sersic': float(all_params['lens_light'][0]['R_sersic']),
            #         'n_sersic': float(all_params['lens_light'][0]['n_sersic']),
            #         'e1': float(all_params['lens_light'][0]['e1']),
            #         'e2': float(all_params['lens_light'][0]['e2']),
            #         'center_x': 0.0,
            #         'center_y': 0.0,
            #         'Ie': float(all_params['lens_light'][0]['Ie'])
            #     }
            # ],
            'noise_params': {
                'back_rms': float(all_params['noise_params']['back_rms']),
                'exp_time': float(all_params['noise_params']['exp_time'])
            }
        }
        
        with open(os.path.join(folder_path, 'params.json'), 'w') as f:
            json.dump(params_to_save, f, indent=4)
        
        # Compute the phi and q of lens_mass
        lens_mass_e1 = float(all_params['lens_mass'][0]['e1'])
        lens_mass_e2 = float(all_params['lens_mass'][0]['e2'])
        phi_lens, q_lens = util.ellipticity2phi_q(lens_mass_e1, lens_mass_e2)
        phi_lens_deg = float(phi_lens * 180 / np.pi)
        # Constrain the position angle to the range 0-180°
        phi_lens_deg = phi_lens_deg % 180.0
        ell_lens = 1.0 / float(q_lens)
        
        # Compute the phi and q of source_light
        source_light_e1 = float(all_params['source_light'][0]['e1'])
        source_light_e2 = float(all_params['source_light'][0]['e2'])
        phi_source, q_source = util.ellipticity2phi_q(source_light_e1, source_light_e2)
        phi_source_deg = float(phi_source * 180 / np.pi)
        # Constrain the position angle to the range 0-180°
        phi_source_deg = phi_source_deg % 180.0
        ell_source = 1.0 / float(q_source)
        
        # Build the model.json data in the required format
        model_data = {
            "pixscale": 0.05,
            "components": [
                {
                    "plane": "source",
                    "size": float(all_params['source_light'][0]['R_sersic']),
                    "ell": ell_source,
                    "ang": phi_source_deg,
                    "x": float(all_params['source_light'][0]['center_x']),
                    "y": float(all_params['source_light'][0]['center_y']),
                    "n_sersic": float(all_params['source_light'][0]['n_sersic']),
                    "Ie": float(all_params['source_light'][0]['Ie'])
                },
                {
                    "plane": "lens",
                    "theta_e": float(all_params['lens_mass'][0]['theta_E']),
                    "ell": ell_lens,
                    "ang": phi_lens_deg,
                    "x": 0.0,
                    "y": 0.0
                }
            ]
        }
        
        # Save the model.json file
        with open(os.path.join(folder_path, 'model.json'), 'w') as f:
            json.dump(model_data, f, indent=4)
    
    print("Generated 100 sets of simulation data!")

if __name__ == "__main__":
    main()

