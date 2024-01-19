import SimpleITK as sitk
import matplotlib.pyplot as plt
#%matplotlib notebook

import numpy as np
import tifffile
import time


def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} "+ f"= {method.GetMetricValue():10.5f}")


def get_data (high_res_name,low_res_name):

    low_res_path = 'data/'+low_res_name

    high_res_path = 'data/'+high_res_name
    
    fixed_image = sitk.ReadImage(low_res_path, sitk.sitkFloat32)
    print ('fixed_image shape', fixed_image.GetSize())

    moving_image = sitk.ReadImage(high_res_path, sitk.sitkFloat32)
    print ('moving_image shape', moving_image.GetSize())
    
    return fixed_image,moving_image
    
def image_registration(fixed_image,moving_image,saving_name):
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,)
    
    t1 = time.time()
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(2)
    registration_method.SetMetricSamplingPercentage(0.5)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,numberOfIterations=20,
        convergenceMinimumValue=1e-6,convergenceWindowSize=100,)
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    print(f" Iteration: {registration_method.GetOptimizerIteration()}")

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(f" Metric value: {registration_method.GetMetricValue()}")
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print("Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()))

    t2 = time.time()

    print ('registration time = ',round((t2-t1)/60) )
    
    transformed_image = sitk.Resample(moving_image,fixed_image,final_transform,sitk.sitkLinear,0.0,moving_image.GetPixelID(),)
    transformed_image = sitk.GetArrayViewFromImage(transformed_image).astype('uint8')
    
    tifffile.imwrite('output/registered_image'+saving_name, transformed_image)



def main():
    print("Code started !")
    names = [
    ('01_3dCT_B45_bag5_100mm.tif' , '01_Substack (7882-8277)_B45_bag5_.tif'),
    ('02_3dCT_B40_bag12_13_300mm.tif' , '02_Substack (6267-6662)_B40_bag12_13.tif'),
    ('03_3dCT_B40_bag22_23_300mmLUT.tif' , '03_Substack (6241-6635)_B40_bag22_23.tif'),
    ('04_3dCT_B40_bag32_33_200mm_Substack(396-1440).tif' , '04_Substack (7285-7546)_B40_bag32_33.tif'),
    ('05_3dCT_B40_bag46_47_300mm.tif' , '05_Substack (6292-6686)_B40_bag46_47.tif'),
    ('06_3dCT_B40_bag56_57_100mm.tif' , '06_Substack (8055-8449)_B40_bag56_57.tif'),
    ('07_3dCT_B40_bag66_67_400mm.tif' , '07_Substack (5525-5920)_B40_bag66_67.tif'),
    ('08_3dCT_B40_bag86_87_640mm.tif' , '08_Substack (3424-3818)_B40_bag86_87.tif'),
    ('09_3dCT_B40_bag96_97_675mm.tif' , '09_Substack (3098-3490)_B40_bag96_97.tif'),
    ('10_3dCT_B40_bag108_109_538mm.tif' , '10_Substack (4268-4663)_B40_bag108_109.tif'),
    ('11_3dCT_B40_bag126_127_1002mm.tif' , '11_Substack (305-698)_B40_bag126_127.tif')
    ]
    
    
    for n in names:
        fixed_image,moving_image = get_data(high_res_name=n[0], low_res_name=n[1])
        image_registration(fixed_image,moving_image,saving_name=n[0])

if __name__ == "__main__":
    main()
