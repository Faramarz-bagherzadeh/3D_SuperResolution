import tifffile
import numpy as np
import skimage
from sklearn.cluster import KMeans
import pumapy as puma
import pandas as pd
from skimage.transform import downscale_local_mean


def puma_permeability (binary):
        
    binary = downscale_local_mean(binary, (2,2,2))
    binary[binary>0.5] = 1
    binary[binary<=0.5] = 0
    ws = puma.Workspace.from_array(binary.copy())
    ws.voxel_length = 128e-6
    
    filled = puma.fill_closed_pores(ws, (1, 1), 1, return_pores=False)

    #ws.binarize(0.5)
    #print(f"Shape of workspace: {ws.matrix.shape}")
    #keff1, u_x1, p_x1, u_y1, p_y1, u_z1, p_z1
    try:
        keff = puma.compute_permeability(filled, solid_cutoff=(1, 1),solver_type='minres',
                                                          direction='xyz', tol=1e-8,
                                                          maxiter=2000,display_iter=True,
                                                          matrix_free=False, precondition=False,
                                                          output_fields=False)
        
        print ('permeability done')

    except:
        print ('Not Permeabale !!!')
        return (0,0,0)

    return keff #Effective permeability tensor


def puma_specific_area(binary):
    ws = puma.Workspace.from_array(binary.copy())
    ws.voxel_length = 60e-6
    # print(f"Shape of workspace: {ws.matrix.shape}")
    area_us, specific_area_us = puma.compute_surface_area(ws, cutoff=(1, 1))

    return specific_area_us


def puma_mean_intercept_length(binary):
    ws = puma.Workspace.from_array(binary.copy())
    ws.voxel_length = 60e-6
    mil = puma.compute_mean_intercept_length(ws, void_cutoff=(1, 1))
    mil = np.array(mil)
    mil = mil * 1000 # change unit to mm
    mil = np.round(mil, 2)
    return mil


def puma_tortuosity (binary):
    print ('Tortousity started !') 
    
    binary = downscale_local_mean(binary, (2,2,2))
    binary[binary>0.5] = 1
    binary[binary<=0.5] = 0
    ws = puma.Workspace.from_array(binary.copy())
    ws.voxel_length = 128e-6
    #print(f"Shape of workspace: {ws.matrix.shape}")
    st = 'cg' #'bicgstab'#'direct' #
    mf = False
    n_eff_x, Deff_x, poro, C_x = puma.compute_continuum_tortuosity(ws, (0,0), 'x', side_bc='s', tolerance=1e-4, solver_type=st, matrix_free = mf)
    n_eff_y, Deff_y, poro, C_y = puma.compute_continuum_tortuosity(ws, (0,0), 'y', side_bc='s', tolerance=1e-4, solver_type=st, matrix_free = mf)
    n_eff_z, Deff_z, poro, C_z = puma.compute_continuum_tortuosity(ws, (0,0), 'z', side_bc='s', tolerance=1e-4, solver_type=st, matrix_free = mf)
    inside_air = str((round(n_eff_x[0],1), round(n_eff_y[1],1) , round(n_eff_z[2],1)))

    n_eff_x, Deff_x, poro, C_x = puma.compute_continuum_tortuosity(ws, (1,1), 'x', side_bc='s', tolerance=1e-4, solver_type=st, matrix_free = mf)
    n_eff_y, Deff_y, poro, C_y = puma.compute_continuum_tortuosity(ws, (1,1), 'y', side_bc='s', tolerance=1e-4, solver_type=st, matrix_free = mf)
    n_eff_z, Deff_z, poro, C_z = puma.compute_continuum_tortuosity(ws, (1,1), 'z', side_bc='s', tolerance=1e-4, solver_type=st, matrix_free = mf)
    inside_ice = str((round(n_eff_x[0],1), round(n_eff_y[1],1) , round(n_eff_z[2],1)))  
    
    
    print ('Tortousity done')
   # poro is the porosity of the material
   # n_eff is the effective tortuosity factor
   # C_x is the computed field vector
    return inside_air , inside_ice


def puma_orientations(binary):
    ors = puma.Workspace.from_array(binary.copy())
    # print(f"Shape of workspace: {ors.matrix.shape}")
    puma.compute_orientation_st(ors, cutoff=(1, 1), sigma=1.4, rho=0.7, edt=True)
    ors_z = ors.orientation[:, :, :, 0].mean()
    ors_y = ors.orientation[:, :, :, 1].mean()
    ors_x = ors.orientation[:, :, :, 2].mean()

    return round(ors_x, 2), round(ors_y, 2), round(ors_z, 2)

def binary_seg_kMeans(img):
    binary = np.zeros_like(img)
    pixels = img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    thresh = (centers[0] + centers[1])/2
    binary[img > thresh] = 1
    return binary

def contrast_stretching(input_image):
    #Contrast stretching
    #Dropping extreems (artifacts)
    p2, p98 = np.percentile(input_image, (2, 98))
    stretched_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
    return stretched_image.astype('uint8')


def read_and_preprocessing(path, s):
    img = tifffile.imread(path)
    img = img[s[0]:s[1], s[2]:s[3], s[4]:s[5]]
    img = contrast_stretching(img).astype('uint8')

    return img

def manual_binary(img, thresh):
    output = np.zeros_like(img)
    output[img>thresh] = 1
    return output
    

def read_binary_weighted(path,s):
    from scipy import ndimage
    img = tifffile.imread(path)
    scale = 2.0
    img = ndimage.zoom(img,scale, order = 1, prefilter=False, grid_mode=False)
    img = img[s[0]:s[1],s[2]:s[3],s[4]:s[5]]
    out = np.zeros_like(img)
    out[img>125] = 1

    return out

def microstructure_metric(img):
    ice_density = 0.92  # (g/cm³)
    pixel_volume = (0.006) ** 3  # cm³
    voxel = 0.006  # cm
    volume = img.shape[0]*img.shape[1]*img.shape[2]*pixel_volume

    relative_density = round(len(img[img > 0]) / img.size, 2)
    density = relative_density * ice_density

    porosity = round(len(img[img == 0]) * 100 / img.size, 2)

    #verts, faces, _, _ = skimage.measure.marching_cubes(img, level=0)
    #surface_area = skimage.measure.mesh_surface_area(verts, faces)
    #SSA = round((surface_area * voxel * voxel), 2)  # mm^2
    SSA = puma_specific_area(img)
    # For 3D objects, the Euler number is obtained as the number of objects
    # plus the number of holes, minus the number of tunnels, or loops.
    euler = skimage.measure.euler_number(img, connectivity=1)/volume

    return {'density': density, 'porosity': porosity, 'SSA': SSA, 'euler': euler}

def main ():


    snow_files = {
                  'High-res': 'data/registered_image02_3dCT_B40_bag12_13_300mm.tif',
                  'Bicubic' : 'data/02_Substack (6267-6662)_B40_bag12_13.tif',
                  'Bicubic_weight' : 'model_output/02_Seg_Weight_Substack (6267-6662)_B40_bag12_13.tif',
                  'SRCNN'   : 'model_output/02_Substack_predictions_SRCNN_.tif',
                  'DCSRN'   : 'model_output/02_Substack_predictions_DCSRN_.tif',
                  'SRUnet'  : 'model_output/02_Substack_predictions_SRUnet_.tif',
                  'SRResnet': 'model_output/02_Substack_predictions_SRResnet_.tif'
                 }
    firn_files = {
                  'High-res': 'data/registered_image06_3dCT_B40_bag56_57_100mm.tif',
                  'Bicubic' : 'data/06_Substack (8055-8449)_B40_bag56_57.tif',
                  'Bicubic_weight' : 'model_output/06_Seg_Weight_Substack (8055-8449)_B40_bag_56_57.tif',
                  'SRCNN'   : 'model_output/06_Substack_predictions_SRCNN_.tif',
                  'DCSRN'   : 'model_output/06_Substack_predictions_DCSRN_.tif',
                  'SRUnet'  : 'model_output/06_Substack_predictions_SRUnet_.tif',
                  'SRResnet': 'model_output/06_Substack_predictions_SRResnet_.tif'
                 }
    ice_files = {
                  'High-res': 'data/registered_image10_3dCT_B40_bag108_109_538mm.tif',
                  'Bicubic' : 'data/10_Substack (4268-4663)_B40_bag108_109.tif',
                  'Bicubic_manual' : 'data/10_Substack (4268-4663)_B40_bag108_109.tif',
                  'Bicubic_weight' : 'model_output/10_Seg_Weight_Substack (4268-4663)_B40_bag108_109.tif',
                  'SRCNN'   : 'model_output/10_Substack_predictions_SRCNN_.tif',
                  'DCSRN'   : 'model_output/10_Substack_predictions_DCSRN_.tif',
                  'SRUnet'  : 'model_output/10_Substack_predictions_SRUnet_.tif',
                  'SRResnet': 'model_output/10_Substack_predictions_SRResnet_.tif'
                }
    snow_slice = [10,410,800,1200,200,600]
    firn_slice = [10,410,600,1000,1200,1600]
    ice_slice= [10,410,200,600,800,1200]

    microstructur_result = pd.DataFrame(columns=['type','name','density','porosity','SSA','euler','mil',
                                                  'orientation','tortuosity_a','tortuosity_i','permeability'])
    row_number= 0
    for num, f in enumerate([snow_files, firn_files, ice_files]):
        type_ = ['snow', 'firn', 'ice'][num]
        print(type_)
        slice_ = [snow_slice, firn_slice, ice_slice][num]
        for name in f.keys():
            row_number += 1 
            print(name)
            print(row_number)
            if name == 'Bicubic_weight':
                data = read_binary_weighted(f[name], slice_)
            else:
                data = read_and_preprocessing(f[name], slice_)
            
            # An exceptional way of segmentation for Bicubic is used
            if name == 'Bicubic_manual':
                # the threshhold is calculated in such was to have same porosity as HR data
                print ('manual threshholding started !!!!')
                data= manual_binary(data, 85) 
                
            elif name == 'Bicubic_weight':
                data = data
                
            else:
                print ('seg with k-means')
                data = binary_seg_kMeans(data)

            

            microstructur_result.at[row_number, 'type'] = type_
            microstructur_result.at[row_number, 'name'] = name
            mm = microstructure_metric(data)
            microstructur_result.at[row_number, 'density'] = round(mm['density'], 2)
            microstructur_result.at[row_number, 'porosity'] = round(mm['porosity'], 2)
            microstructur_result.at[row_number, 'SSA'] = round(mm['SSA'], 0)
            microstructur_result.at[row_number, 'euler'] = round(mm['euler'], 0)
            microstructur_result.at[row_number, 'mil'] = str(puma_mean_intercept_length(data))
            
            microstructur_result.at[row_number, 'permeability'] = str(puma_permeability(data))
            microstructur_result.at[row_number, 'orientation'] = str(puma_orientations(data))
            microstructur_result.at[row_number, 'tortuosity_a'], microstructur_result.at[row_number, 'tortuosity_i'] = puma_tortuosity(data)

            microstructur_result.to_csv('microstructure_metrics.csv' , index=False)
        

if __name__ == "__main__":
    main()

