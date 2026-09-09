import os
import sys
import numpy as np
import torch
from astropy.table import Table
from astropy.io import fits
from PIL import Image
import io

# Add GGSL_UNet to the path so that its model can be used
GGSL_UNET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../GGSL_UNet'))
if GGSL_UNET_PATH not in sys.path:
    sys.path.append(GGSL_UNET_PATH)

# Import only the required model components
from GGSL_UNet import diffusion_model
from GGSL_UNet.plot_image import calculate_real_chi

def convert_ordinary_fits_to_model_format(fits_file, output_fits_path=None):
    '''
    Convert a regular FITS file object into the special format required by the GGL_UNet model
    Parameters:
        fits_file: FITS file object (e.g. UploadFile.file or BytesIO)
        output_fits_path: output path of the model-format FITS file, defaults to None (auto-generated)
    Returns:
        output_fits_path: path of the converted FITS file
        original_min: minimum value of the original image
        original_max: maximum value of the original image
    '''
    # Read the file object
    if hasattr(fits_file, "read"):
        fits_file.seek(0)
        hdul = fits.open(io.BytesIO(fits_file.read()))
    elif isinstance(fits_file, fits.HDUList):
        hdul = fits_file
    else:
        hdul = fits.open(fits_file)
    
    try:
        # Ensure the HDU list is not empty
        if not hdul:
            raise ValueError(f"The FITS file does not contain any data units")
        
        # Check whether the primary HDU contains data
        if hdul[0].data is None:
            raise ValueError(f"The primary HDU of the FITS file is empty")
        
        # Assume the data is in the primary HDU
        image_data = hdul[0].data.astype(np.float32)
        
        # Save the minimum and maximum values of the original image
        original_min = image_data.min()
        original_max = image_data.max()
        # Check the image dimensions and adjust accordingly
        if len(image_data.shape) == 2:  # single-channel image
            # Resize to 64x64
            img_pil = Image.fromarray(image_data)
            img_pil = img_pil.resize((64, 64), Image.LANCZOS)
            img_np = np.array(img_pil).astype(np.float32)
            
            # Normalize to the [0, 1] range
            if img_np.max() - img_np.min() > 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Create a 3-channel RGB image (duplicate the single channel across three channels)
            img_rgb = np.stack([img_np, img_np, img_np], axis=2)  # [64, 64, 3]
            
            # Generate the 4th channel (grayscale average)
            gray = img_rgb.mean(axis=2, keepdims=True)  # [64, 64, 1]
            img_np_4ch = np.concatenate([img_rgb, gray], axis=2)  # [64, 64, 4]
            
            # Convert to [1, 4, 64, 64] format
            img_np_4ch = np.transpose(img_np_4ch, (2, 0, 1))   # [4, 64, 64]
            img_np_4ch = img_np_4ch[np.newaxis, ...]           # [1, 4, 64, 64]
            
        elif len(image_data.shape) == 3 and image_data.shape[2] == 3:  # RGB image
            # Resize to 64x64
            img_pil = Image.fromarray((image_data * 255).astype(np.uint8))
            img_pil = img_pil.resize((64, 64), Image.LANCZOS)
            img_np = np.array(img_pil).astype(np.float32) / 255.0  # [64, 64, 3]
            
            # Normalize to the [0, 1] range
            if img_np.max() - img_np.min() > 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Generate the 4th channel (grayscale average)
            gray = img_np.mean(axis=2, keepdims=True)  # [64, 64, 1]
            img_np_4ch = np.concatenate([img_np, gray], axis=2)  # [64, 64, 4]
            
            # Convert to [1, 4, 64, 64] format
            img_np_4ch = np.transpose(img_np_4ch, (2, 0, 1))   # [4, 64, 64]
            img_np_4ch = img_np_4ch[np.newaxis, ...]           # [1, 4, 64, 64]
        
        else:
            raise ValueError(f"Unsupported image format, shape: {image_data.shape}")
        
        # Generate weight, RA, DEC
        weight = np.ones_like(img_np_4ch)
        RA = np.array([180.0])
        DEC = np.array([0.0])
        
        # Build and save as a FITS table
        table = Table({
            'image': img_np_4ch,
            'weight': weight,
            'RA': RA,
            'DEC': DEC
        })
        
        # Generate a temporary file name
        if output_fits_path is None:
            output_fits_path = "temp_model_format.fits"
        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the file
        table.write(output_fits_path, overwrite=True)
        print(f"Successfully converted the regular FITS file to the model-required format: {output_fits_path}")
        
        return output_fits_path, original_min, original_max
    
    except Exception as e:
        print(f"Error converting FITS file format: {str(e)}")
        raise

def image_preprocess(data, norm, channel=1):
    '''
    # Linearly transform the image and parameters (reversible)
    # Supports both single images and batches of images
    '''
    data = data.astype(np.float32)
    data = data * norm
    data = torch.from_numpy(data) # Convert to tensor

    # Determine the input dimensions
    if data.ndim == 2:  # [H, W]
        data = data.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif data.ndim == 3:
        if data.shape[0] == channel:  # [C, H, W]
            data = data.unsqueeze(0)  # [1, C, H, W]
        else:  # [N, H, W]
            data = data.unsqueeze(1)  # [N, 1, H, W]
    # Leave unchanged if already [N, C, H, W]

    return data

def extract_background_from_fits(fits_file_path, ggsl_unet_dir=None, original_shape=None):
    '''
    Extract the background image from a model-format FITS file path
    Parameters:
        fits_file_path: path to the FITS file in the format required by the model
        ggsl_unet_dir: path to the GGSL_UNet directory, defaults to None (auto-detected)
        original_shape: shape of the original FITS file, defaults to None
    Returns:
        background_array: extracted background image array, with the same shape as the original FITS file
        pred_data: raw prediction result data, used for computing the chi-square value
    '''
    # Validate parameters
    if not os.path.exists(fits_file_path):
        raise FileNotFoundError(f"FITS file not found: {fits_file_path}")
    
    # Determine the GGSL_UNet directory
    if ggsl_unet_dir is None:
        ggsl_unet_dir = GGSL_UNET_PATH
    
    # Save the original working directory
    orig_cwd = os.getcwd()
    
    try:
        # Switch to the GGSL_UNet directory to ensure the model loads its resources correctly
        os.chdir(ggsl_unet_dir)
        
        # Ensure the results directory exists
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Initialize the GGL_UNet model
        ggl_unet = diffusion_model.GGL_UNet()
        ggl_unet.cfg['train_set'] = 'color_images.fits'
        ggl_unet.Init()
        ggl_unet.load_training_data(frac=0)  # Do not load training data
        # Force the model to run on CPU to avoid device mismatch issues
        ggl_unet.device = torch.device('cpu')
        
        # Read the FITS file data
        data_lens = Table.read(fits_file_path)
        # Validate the data format
        if 'image' not in data_lens.colnames:
            raise ValueError(f"FITS file {fits_file_path} does not contain an 'image' column")
        
        # Extract the image data and weight data
        color_images = data_lens['image']
        # Check and extract the optional weight, RA and DEC data
        weigh_image = data_lens['weight'] if 'weight' in data_lens.colnames else []
        RA = data_lens['RA'] if 'RA' in data_lens.colnames else []
        DEC = data_lens['DEC'] if 'DEC' in data_lens.colnames else []
        
        # Ensure color_images is not None
        if color_images is None:
            raise ValueError(f"The 'image' column data in FITS file {fits_file_path} is None")
        
        # Handle byte-order issues to ensure the data is passed to the model in the correct format
        if isinstance(color_images, np.ndarray) and color_images.dtype.byteorder != '=':
            # Convert to native byte order
            color_images = color_images.astype(color_images.dtype.newbyteorder('='))
        
        # Use the model to predict the background
        # Generate a temporary save name
        temp_save_name = f'pred_real_temp_{os.getpid()}'
        ggl_unet.pred_real(
            color_images, 
            weight=weigh_image, 
            RA=RA, 
            DEC=DEC, 
            timestep=25, 
            load_model=True, 
            save_name=temp_save_name
        )
        
        # Read the prediction results
        results_fits = f'results/GGL_UNet_diffusion_model_{temp_save_name}.fits'
        if not os.path.exists(results_fits):
            raise FileNotFoundError(f"Model prediction result file was not generated: {results_fits}")
            
        pred_data = Table.read(results_fits)
        
        # Validate the prediction result format
        required_columns = ['input', 'pred_back', 'pred_fore', 'pred_noise', 'weight', 'norm']
        for col in required_columns:
            if col not in pred_data.colnames:
                raise ValueError(f"The prediction result file does not contain the '{col}' column")
        
        # If no original shape is provided, default to 64x64 (the model's output shape)
        if original_shape is None:
            original_shape = (64, 64)
        
        # Extract the background image data
        pred_back = image_preprocess(pred_data['pred_back'], channel=4, norm=1)
        pred_back = torch.nn.functional.interpolate(pred_back, size=original_shape, mode='bilinear', align_corners=False)
        pred_back = pred_back.mean(dim=1, keepdim=True)  # Convert to grayscale
        pred_back = pred_back.squeeze().numpy()
        
        # Delete the temporary file
        if os.path.exists(results_fits):
            try:
                os.remove(results_fits)
            except:
                print(f"Warning: failed to delete temporary file {results_fits}")
        
        # Ensure the returned shape is correct
        expected_shape = original_shape
        for name, arr in zip(['pred_back'], [pred_back]):
            if arr.shape != expected_shape:
                raise ValueError(f"Incorrect shape for {name}: {arr.shape}, expected {expected_shape}")
            
        return pred_back, pred_data
        
    except FileNotFoundError as e:
        print(f"File error: {str(e)}")
        raise
    except ValueError as e:
        print(f"Data format error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error extracting background: {str(e)}")
        raise
    finally:
        # Switch back to the original working directory
        os.chdir(orig_cwd)

def process_ordinary_fits_and_extract_background(fits_file, output_fits_path=None, save_background=False, temp_dir=None):
    """
    Full pipeline: process a FITS file object directly, extract the background and display it
    Parameters:
        fits_file: FITS file object (e.g. UploadFile.file or BytesIO)
        output_fits_path: path of the background FITS file to save, defaults to None (do not save)
        save_background: whether to save the extracted background as a FITS file
        temp_dir: directory for temporary files, defaults to None (use the current directory)
    Returns:
        background: extracted background image array, with the same shape as the original FITS file
        pred_data: raw prediction result data, used for computing the chi-square value
    """
    
    try:
        # Determine the temporary directory
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save the shape of the original FITS file
        original_shape = None
        if hasattr(fits_file, "read"):
            fits_file.seek(0)
            hdul = fits.open(io.BytesIO(fits_file.read()))
            original_shape = hdul[0].data.shape
            fits_file.seek(0)  # Reset the file pointer
        elif isinstance(fits_file, str):
            hdul = fits.open(fits_file)
            original_shape = hdul[0].data.shape
        
        # Step 1: convert the file object to the format required by the model
        model_format_fits_path = os.path.join(temp_dir, f"temp_model_format_{os.getpid()}.fits")
        model_format_fits_path, original_min, original_max = convert_ordinary_fits_to_model_format(fits_file, model_format_fits_path)
        
        # Step 2: extract the background and other prediction results from the converted file
        background, pred_data = extract_background_from_fits(model_format_fits_path, original_shape=original_shape)
        
        # Step 3: restore the original flux and ensure non-negativity
        # Compute the original range
        original_range = original_max - original_min
        # Restore to the original range
        background = background * original_range + original_min
        # Ensure all values are non-negative (physically, flux cannot be negative)
        background = np.clip(background, 0, None)
        
        # Step 4: optionally save the background image as a FITS file
        if save_background and output_fits_path:
            output_dir = os.path.dirname(output_fits_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            hdu = fits.PrimaryHDU(data=background)
            hdul = fits.HDUList([hdu])
            hdul.writeto(output_fits_path, overwrite=True)
            print(f"Saved the extracted background as a FITS file: {output_fits_path}")
        
        # Clean up the temporary file
        if os.path.exists(model_format_fits_path):
            try:
                os.remove(model_format_fits_path)
            except:
                print(f"Warning: failed to delete temporary file {model_format_fits_path}")
                
        return background, pred_data
    except Exception as e:
        print(f"Error processing FITS file and extracting background: {str(e)}")
        raise