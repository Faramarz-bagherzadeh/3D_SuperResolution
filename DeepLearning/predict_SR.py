import numpy as np
import tifffile
import torch
import skimage
from patchify import patchify, unpatchify




if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {device_count}")
    for i in range(device_count):
        device = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device}")
else:
    print("CUDA is not available on this machine.")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def contrast_stretching(img):
    # Contrast stretching
    # Dropping extreems (artifacts)
    batch = 32
    print ('performed on a batch of = ', batch)
    for i in range ((img.shape[0]//batch)+1):
        if (i+1)*batch <img.shape[0]:
            input_image = img[i*batch:(i+1)*batch]
            p2, p98 = np.percentile(input_image, (2, 98))
            img[i*batch:(i+1)*batch] = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
        else:
            input_image = img[i*batch:]
            p2, p98 = np.percentile(input_image, (2, 98))
            img[i*batch:] = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
        print(f"Progress: {round(((i+1)*batch)*100/img.shape[0])} %", end='\r')
    return img


def reshape_to_power_of_2(data, patch_size):
    # Check if the shape of the image in each dimension is a power of 2
    original_shape = data.shape
    
    padding_1 = [(patch_size, patch_size),(patch_size, patch_size),(patch_size, patch_size)]
    data = np.pad(data, padding_1, mode='constant', constant_values=0)
    
    new_shape = [2**int(np.ceil(np.log2(dim))) for dim in data.shape]
    padding_2 = [(0, new_dim - old_dim) for old_dim, new_dim in zip(data.shape, new_shape)]
    data = np.pad(data, padding_2, mode='constant', constant_values=0)
    
    print ('original_shape =',original_shape)
    print ('padding_1 = ',padding_1)
    print ('padding_2 = ',padding_2)
    print ('data shape after padding = ', data.shape)
    
    return original_shape, padding_1,padding_2, data


def patchyfy_img(img, ps, step):
    padded_shape = img.shape
    img = patchify(img,(ps, ps, ps) ,  step=step )
    patched_shape = img.shape
    img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2],ps,ps,ps )
    return padded_shape, patched_shape,img

def build_original_image(img,patched_shape, padded_shape, original_shape, padding_1, padding_2):
    img = img.reshape(patched_shape)
    print (img.shape)
    img = unpatchify(img, padded_shape)
    print (img.shape)
    img = img[:img.shape[0]-padding_2[0][1], :img.shape[1]-padding_2[1][1], :img.shape[2]-padding_2[2][1]]
    print (img.shape)
    img = img[32:-32,32:-32,32:-32]
    print (img.shape)
    return img

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def predict(model,data,patch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    prediction = np.zeros((data.shape[0],patch_size,patch_size,patch_size))
    print ('number of cpus',torch.get_num_threads())
    torch.set_num_threads(120)
    print ('number of cpus changed to ',torch.get_num_threads())

    
    for i in range(data.shape[0]):
        input_= torch.from_numpy(data[i]).unsqueeze(0).unsqueeze(0).float()
        input_ = input_/255 #normalization
        input_ = input_.to(device)
        output_= model(input_).cpu().detach().numpy()
        output_ = output_[0,0]*255 # back to real scale
        output_ = output_.astype('uint8')
        prediction[i]=output_
        print(f"Progress: {round(i*100/data.shape[0])} %", end='\r')
    return prediction

def main(data,name):
    patch_size = 64
    
    data = contrast_stretching(data)
    original_shape, padding_1,padding_2, data = reshape_to_power_of_2(data,patch_size)
    padded_shape, patched_shape, data = patchyfy_img(data,ps=2*patch_size,step=patch_size)
    
    import model_SRCNN
    model = model_SRCNN.SRCNN()
    load_checkpoint(torch.load("my_SRCNN_checkpoint.pth.tar",map_location=torch.device(device) ), model)
   # import UNET
   # model = UNET.UNet(1,1)
   # load_checkpoint(torch.load("my_SRUNET_checkpoint.pth.tar",map_location=torch.device('cpu') ), model) 
    
    #import DCSRN
    #model = DCSRN.DCSRN(1)
    #load_checkpoint(torch.load("my_DCSRN_checkpoint.pth.tar",map_location=torch.device('cpu') ), model)
   # import SRResnet
   # model = SRResnet.SRResNet()
   # load_checkpoint(torch.load("my_SRResnet_checkpoint.pth.tar",map_location=torch.device(device) ), model)
    

    prediction = predict(model,data,patch_size)
    new_paded_shape = np.array(padded_shape)-patch_size
    new_patched_shape = list(patched_shape[:3]) + [64,64,64]

    prediction2 = build_original_image(prediction,new_patched_shape,
                                       new_paded_shape,original_shape,
                                       padding_1, padding_2)
    prediction2 = prediction2.astype('uint8')
    
    tifffile.imwrite('model_output/'+name+'_predictions_SRCNN_.tif', prediction2)
    
    
if __name__ == "__main__":
    
    f=1 # which file to be predicted
    #batch_size = 1 we considered the batch should be 1 for performing predictions
    test_files = [
        '02_Substack (6267-6662)_B40_bag12_13.tif',
        '06_Substack (8055-8449)_B40_bag56_57.tif',
        '10_Substack (4268-4663)_B40_bag108_109.tif']
    
    for f in range (len(test_files)):
        
        data = tifffile.imread('data/'+test_files[f])
        name = test_files[f][:11]
        main(data, name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
