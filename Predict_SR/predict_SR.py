


def contrast_stretching(img):
    # Contrast stretching
    # Dropping extreems (artifacts)
    batch = 64
    print ('performed on a batch of = ', batch)
    for i in range ((img.shape[0]//batch)+1):
        if (i+1)*batch <img.shape[0]:
            input_image = img[i*batch:(i+1)*batch]
            p2, p98 = np.percentile(input_image, (2, 98))
            img[i*batch:(i+1)*batch] = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
            print(f"Progress: {round(((i+1)*batch)*100/img.shape[0])} %", end='\r')

        else:
            input_image = img[i*batch:] 
            if input_image.size > 10:
                p2, p98 = np.percentile(input_image, (2, 98))
                img[i*batch:] = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
    return img


def reshape_to_power_of_2(data, patch_size):
    # Check if the shape of the image in each dimension is a power of 2
    original_shape = data.shape

    padding_1 = [(patch_size, patch_size),(patch_size, patch_size),(patch_size, patch_size)]
    data = np.pad(data, padding_1, mode='constant', constant_values=0)

    new_shape = [2**int(np.ceil(np.log2(dim))) for dim in data.shape]
    padding_2 = [(0, new_dim - old_dim) for old_dim, new_dim in zip(data.shape, new_shape)]
    data = np.pad(data, padding_2, mode='constant', constant_values=0)

    #print ('original_shape =',original_shape)
    #print ('padding_1 = ',padding_1)
    #print ('padding_2 = ',padding_2)
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
    #print (img.shape)
    img = unpatchify(img, padded_shape)
    #print (img.shape)
    img = img[:img.shape[0]-padding_2[0][1], :img.shape[1]-padding_2[1][1], :img.shape[2]-padding_2[2][1]]
    #print (img.shape)
    img = img[32:-32,32:-32,32:-32]
    #print (img.shape)
    return img

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def predict(model,data,patch_size):

    prediction = np.zeros((data.shape[0],patch_size,patch_size,patch_size),dtype='uint8')
    #print ('number of cpus',torch.get_num_threads())
    torch.set_num_threads(120)
    #print ('number of cpus changed to ',torch.get_num_threads())
    non_zero_indices = np.where(np.sum(data, axis=(1,2,3)) != 0)[0]
    batch_size = 8
    loop_len = data[non_zero_indices].shape[0]//batch_size
    for i in range(0,loop_len+1):
        
        if i+1 > loop_len:
            r = [i*batch_size, None] # make sure we read all patches
        else:
            r = [i*batch_size, (i+1)*batch_size]
                
        input_= torch.from_numpy(data[non_zero_indices[r[0]:r[1]]]).unsqueeze(1).float()
        #print ('input sum = ', input_.sum())
        #print ('input_ = ',input_.shape)
        input_ = input_/255 #normalization
        input_ = input_.to(device)
        
        output_= model(input_).cpu().detach().numpy()
        
        #print('shape of output of model',output_.shape)
        output_ = output_[:,0,:,:,:]*255 # back to real scale
        output_ = output_.astype('uint8')
        
        #print('shape of output of model before appending',output_.shape)
        #print ('output sum = ', output_[:,32:-32,32:-32,32:-32].sum())
        prediction[non_zero_indices[r[0]:r[1]]]=output_[:,32:-32,32:-32,32:-32]
        #print(f"Progress: {round(i*100/data.shape[0])} %", end='\r')
    return prediction


def Super_resolution(data,model):
    patch_size = 64

    data = contrast_stretching(data)
    original_shape, padding_1,padding_2, data = reshape_to_power_of_2(data,patch_size)
    padded_shape, patched_shape, data = patchyfy_img(data,ps=2*patch_size,step=patch_size)


    prediction = predict(model,data,patch_size)
    new_paded_shape = np.array(padded_shape)-patch_size
    new_patched_shape = list(patched_shape[:3]) + [64,64,64]

    prediction2 = build_original_image(prediction,new_patched_shape,
                                       new_paded_shape,original_shape,
                                       padding_1, padding_2)
    prediction2 = prediction2.astype('uint8')
    return prediction2



def data_spliting (data,model):
    print ('shape of data before spliting', data.shape)
    scale = 2.0
    output = np.zeros_like(ndimage.zoom(data[:1,:,:],scale, order = 1, prefilter=False, grid_mode=False))
    step = 64

    for s in range (0,data.shape[0],step):
        print ('steps = ',s,s+step)
        if s+step > data.shape[0]:
            img = data[s:,:,:]
        else:
            img = data[s:s+step,:,:]
            
        img = ndimage.zoom(img,scale, order = 1, prefilter=False, grid_mode=False)
        
        output = np.concatenate((output, Super_resolution(img,model)), axis=0) 
        
    return output

    
    
if __name__ == "__main__":
    import numpy as np
    import tifffile
    import torch
    import skimage
    from patchify import patchify, unpatchify
    import glob
    from scipy import ndimage
    import torch.nn as nn
    import time
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {device_count}")
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device}")
    else:
        print("CUDA is not available on this machine.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

   # import model_SRCNN
   # model = model_SRCNN.SRCNN()
   # load_checkpoint(torch.load("my_SRCNN_checkpoint.pth.tar",map_location=torch.device(device) ), model)
   # import UNET
   # model = UNET.UNet(1,1)
   # load_checkpoint(torch.load("my_SRUNET_checkpoint.pth.tar",map_location=torch.device('cpu') ), model)

   # import DCSRN
   # model = DCSRN.DCSRN(1)
   # load_checkpoint(torch.load("my_DCSRN_checkpoint.pth.tar",map_location=torch.device('cpu') ), model)

    import SRResnet
    model = SRResnet.SRResNet()
    load_checkpoint(torch.load("my_SRResnet_checkpoint.pth.tar",map_location=torch.device(device) ), model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##device = 'cpu'
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    
    paths = glob.glob('data/*')
    print(paths)

    
    for f in paths:
        t1 = time.time()
        data = tifffile.imread(f)
        name = f[6:-4]
        print ('*********************************************')
        print ('file name = ', name)
        print ('low res shape = ',data.shape)
        
        output = data_spliting(data, model)
       
        tifffile.imwrite('output/'+name+'_predictions_SRResNet_B_.tif', output)
        #break
    
        t2= time.time()
        print ('Time (h) =', round((t2-t1)/3600))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
