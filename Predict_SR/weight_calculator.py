import cv2
import numpy as np
import tifffile


def get_ice_part(image, thresh1, thresh2, thickness):
    final_mask = np.zeros_like(image)
    for i in range(image.shape[0]):
        img = image[i]
        # Apply a binary threshold to create a binary image
        _, binary = cv2.threshold(img, thresh1, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        mask_1 = np.zeros_like(binary) # for removing tube
        mask_2 = np.zeros_like(binary) # contains ice
        mask_3 = np.zeros_like(binary) # for removing middle part
        #print ('number of contours in first mask',len(contours))
        for cnt in contours[:]:
            area = cv2.contourArea(cnt)
            if area > 100:  # Adjust this threshold based on the size of the ice pieces
                cv2.drawContours(mask_1, [cnt], -1, 1, thickness=thickness)
                cv2.drawContours(mask_2, [cnt], -1, 1, thickness=-1)
        mask_2[mask_1 == 1] =0
        img = mask_2*img
        _, binary = cv2.threshold(img, thresh2, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.dilate(binary, kernel=None, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        #print ('number of contours in second mask',len(contours))
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours[:2]:
            area = cv2.contourArea(cnt)
            if area > 100:  # Adjust this threshold based on the size of the ice pieces
                cv2.drawContours(mask_3, [cnt], -1, 1, thickness=-1)
        final_mask[i] = mask_3
    
    return final_mask



def contrast_stretching(input_image):
    import skimage
    #Contrast stretching
    #Dropping extreems (artifacts)
    p2, p98 = np.percentile(input_image, (2, 98))
    stretched_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
    return stretched_image.astype('uint8')

def drop_layers(mask,n):
    #n is the dropping rate
    accepted_layers = [i for i in range(0, mask.shape[0],n)]
    output = np.zeros_like(mask)
    output[accepted_layers] = mask[accepted_layers]
    return output
            
        

def binary_seg_kMeans(img,mask):
    from sklearn.cluster import KMeans
    from joblib import parallel_backend

    
    binary = np.zeros_like(img)
    mask = drop_layers(mask, 8) # take a sample image every 8 layers
    pixels = img[mask==1].reshape(-1, 1)
    with parallel_backend('threading', n_jobs=-1):
        kmeans = KMeans(n_init=4, n_clusters=2,)
        kmeans.fit(pixels)
        
    centers = kmeans.cluster_centers_
    thresh = (centers[0] + centers[1])/2
    binary[img > thresh] = 1
    
    return binary, round(thresh[0])



def calculate_weight(image, batch,density):
    thresh_list=[]
    ice_pixel_sum =0
    for s in range (0,image.shape[0],batch):
        print ('steps = ',s,s+batch)
        if s+batch > image.shape[0]:
            img = image[s:,:,:]
        else:
            img = image[s:s+batch,:,:]
                
        #print(img.shape)
        img = contrast_stretching(img)
        mask = get_ice_part(img,10,20,30)
        binary, k_thresh = binary_seg_kMeans(img,mask)
        thresh_list.append(k_thresh)
        if s == 0:
            tube = binary[2].sum()
            
        ice_pixel_sum += binary.sum() - tube *img.shape[0]
        weight =  ice_pixel_sum* density * pixel_vol
        
    print ('weight =',weight)
    
    print ('thresh list =', thresh_list)
    
    return weight


if __name__ == "__main__":
    import numpy as np
    import tifffile
    import skimage
    import glob
    import time
    
    path = 'C:/Users/Fabagh001/Desktop/DL/Predict_with_SRRESNET/data/*'

    paths = glob.glob(path)
    #paths = glob.glob('output/*')
    print(paths)

    density = 0.917
    batch = 1000
    #pixel_vol = 0.05818**3 #cm3
    pixel_vol = 0.0119**3 #cm3
    weight_list = []
    for f in paths:
        t1 = time.time()
        data = tifffile.imread(f)
        name = f[6:-4]
        print ('*********************************************')
        print ('file name = ', name)
        print ('shape = ',data.shape)
        w = calculate_weight(data, batch, density)
        weight_list.append((name, w))

        t2= time.time()
        print ('Time (h) =', round((t2-t1)/3600))
        
    print ('weight list = ', weight_list)


















