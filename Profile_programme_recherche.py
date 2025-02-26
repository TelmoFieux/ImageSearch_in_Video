import requests
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import cv2
import timeit
import torch_directml


def has_transparency(img_path):
    img = Image.open(img_path)
    if img.info.get("transparency", None) is not None:
        return True
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False

# Function to check if the image has an accepted aspect ratio
def has_accepted_aspect_ratio(image_path):
    # Define accepted aspect ratios and a tolerance for floating-point precision
    accepted_aspect_ratios = [(16, 9), (4, 3)]
    tolerance = 0.01  # Define a small tolerance for floating-point comparisons

    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        return False

    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Check if the aspect ratio matches any of the accepted ratios within the tolerance
    for w, h in accepted_aspect_ratios:
        target_ratio = w / h
        if abs(aspect_ratio - target_ratio) < tolerance:
            return True  # The image matches one of the accepted aspect ratios

    return False  # Reject if no match found

def filter_image():
    garbage =[]
    image_names = list(glob.glob('./*.jpg'))
    print("Number of images unfiltered:", len(image_names))
    encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=1, convert_to_tensor=True, show_progress_bar=True,device=device)

    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = len(image_names)

    print('Finding duplicate images...')
    duplicates = [image for image in processed_images if image[0] >= filtering_treshold]

    # On met a la poubelle les image trop similaire
    for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
        if image_names[image_id1] not in garbage:
            garbage.append(image_names[image_id1])
        print("\nScore: {:.3f}%".format(score * 100))
        print(image_names[image_id1],flush=True)
        print(image_names[image_id2],flush=True)

    #on met a la poubelle les image transparentes
    for k in image_names :
        if has_transparency(k) and k not in garbage:
            garbage.append(k)
        elif k not in garbage and has_accepted_aspect_ratio(k)==False:
            garbage.append(k)
        

    #On supprime toute les image non necessaire
    for k in garbage :
        try:
            os.remove(k)
            print(f"{k} has been deleted successfully.")
            image_names.remove(k)
        except FileNotFoundError:
            print(f"{k} does not exist.")
        except PermissionError:
            print(f"Permission denied to delete {k}.")
        except Exception as e:
            print(f"Error occurred while deleting file: {e}")
    print("Number of images filtered:", len(image_names))

def encode_les_images():
    # Get the list of images in the directory
    image_names = list(glob.glob('./*.jpg'))
    # Encode all other images in the directory, and move them to DirectML device
    encoded_images = model.encode([Image.open(filepath) for filepath in image_names], 
                                  batch_size=1, 
                                  convert_to_tensor=True, 
                                  show_progress_bar=True, 
                                  device=device)
    return encoded_images
        
#Cette fonction calcule la similarité entre toute les pairs le nombre de comparaison est donc exponentielle ~1sec pour cheque 1 frame avec 7 image
#!Batch : 1.27 it/s avec 5 image en moyenne cpu batch size = 4
#!Batch : 5.4 it/s avec 5 image en moyenne cpu batch size = 2
#!Batch : 9 it/s {entre 7 et 10} avec 5 image en moyenne gpu batch size = 2 | Batches 3/3
#!Batch : 15 it/s {entre 11 et 14.5} avec 5 image en moyenne gpu batch size = 2 | Batches 5/5

def find_matching_imagesV1(frame_path):
    image_names = list(glob.glob('./*.jpg'))
                    
    # Next we compute the embeddings
    # To encode an image, you can use the following code:
    encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=1, convert_to_tensor=True, show_progress_bar=True,device=device)
    # Now we run the clustering algorithm. This function compares images aganist 
    # all other images and returns a list with the pairs that have the highest 
    # cosine similarity score
    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = len(image_names)
    # =================
    # NEAR DUPLICATES
    # =================
    print('Finding near duplicate images...')

    for score, image_id1, image_id2 in processed_images[0:NUM_SIMILAR_IMAGES]:
        if score*100 < matchin_images_treshold:
            return 0
        print(image_names[image_id1])
        print(image_names[image_id2])
        if image_names[image_id1]==frame_path or image_names[image_id2]==frame_path:
            return (score * 100)
    return 0

#!Batch : 3 it/s avec 5 image en moyenne avec batch size =4 batches : 1/1 avec gpu
#!Batch : 6 it/s {entre 5 et 7 it/s} avec 5 image en moyenne avec batch size =2 batches 2/2 avec gpu
#!Batch : 12 it/s avec 5 image en moyenne avec batch size =1 batches 4/4 avec gpu
def find_matching_imagesV2(frame_path,encoded_images):
    # Encode the image you want to compare, and move it to DirectML device
    frame_image_embedding = model.encode([Image.open(frame_path)], batch_size=1 ,convert_to_tensor=True, device=device)

    # Compute cosine similarities between the frame image and the other images (on AMD GPU)
    similarities = util.cos_sim(frame_image_embedding, encoded_images)

    # Go through the similarity scores
    max_list=[]
    for i, score in enumerate(similarities[0]):
        score_percentage = score.item() * 100  # Convert the similarity score to percentage
        max_list.append(score_percentage)
    return max(max_list)

def get_images(query,api_key,cx,number_of_request):
    for k in range(number_of_request):
        # The API endpoint
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}&searchType=image&imgType=photo&hl=en&num=10&start={(k*10)+1}"

        # Making the HTTP GET request
        response = requests.get(url)

        # Parsing the JSON response
        results = response.json()

        filename="downloaded_image"
        chiffre=k*10

        # Print the image results
        for item in results.get("items", []):
            image_url=item['link']
            response = requests.get(image_url)
            real_filename=filename+str(chiffre)+".jpg"
            chiffre=chiffre+1
            # Check if the request was successful
            if response.status_code == 200:
                # Open a file in binary write mode and write the content
                with open(real_filename, "wb") as file:
                    file.write(response.content)
                    print("Image downloaded successfully!")
            else:
                print("Failed to download the image.")



def scan_video(videopath):
    # Open the video file
    video = cv2.VideoCapture(videopath)

    # Initialize variables
    found_timestamps = {75:[],80:[],85:[],90:[],100:[],"acceleration":0,"décélération":0}
    video_lenght = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    encode_images = encode_les_images()
    seuil=[75,80,85,90,100]
    prev_prob=0
    nb_probs= 0

    # Loop through each frame of the video
    for i in range(0,video_lenght,frame_gap):

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()

        if not ret:
            break
        # Convert frame to grayscale
        if cv2.imwrite('extracted_frame.jpg', frame) == False:
            cv2.imwrite('extracted_frame.jpg', frame)

        # Perform template matching
        prob = find_matching_imagesV2('.\extracted_frame.jpg',encode_images)

        # If match is above threshold, record the timestamp

        nb_probs+=1
        for k in seuil:
            if prob<k:
                found_timestamps[k].append(prob)
                if prob!=0 and prob>prev_prob and prev_prob < k-5:
                    found_timestamps["décélération"]+=1
                elif prob!=0 and prob<prev_prob and prev_prob > k:
                    found_timestamps["acceleration"]+=1

                break
        prev_prob = prob
        
    print('found_timestamps :',found_timestamps,"\n")
    video.release()
    os.remove("extracted_frame.jpg")
    frame_traite=(video_lenght/60)/(frame_gap/60)
    print("video lenght : ", (video_lenght/60))
    print("frame traité : ", frame_traite )
    print("nb_probs : ",nb_probs)

    # Output the found timestamps
    print("Pour la recherche " + query + " sur la vidéo " + VIDEO_SCANNED + " avec 5 sec de frame gap: \n")
    print("- " + str((len(found_timestamps[75])/nb_probs)*100) + "'%' d'image entre 0'%' et 75'%' de similarité")
    print("- " + str((len(found_timestamps[80])/nb_probs)*100) + "'%' d'image entre 75'%' et 80'%' de similarité")
    print("- " + str((len(found_timestamps[85])/nb_probs)*100) + "'%' d'image entre 80'%' et 85'%' de similarité")
    print("- " + str((len(found_timestamps[90])/nb_probs)*100) + "'%' d'image entre 85'%' et 90'%' de similarité")
    print("- " + str((len(found_timestamps[100])/nb_probs)*100) + "'%' d'image entre 90'%' et 100'%' de similarité")
    print("- " +str(found_timestamps["acceleration"]) + " accélérations")
    print("- " +str(found_timestamps["décélération"]) + " décélération")


#? KEY PARAMETERS

VIDEO_SCANNED = "Ratchet_et_clank_2_part_7_CRF.mp4"
# Your API key
api_key = ""
# Your Custom Search Engine ID
cx = ""
# The search query
query = "Ratchet and Clank ps2 hProydropack"
# Number of request (10 images per request)
num_requests = 2
# seuil d'acceptation pour la similarité entre image valeur comprise entre 0 et 100
# 85 pour le modele complet semble bien et 88 pour le modele half semble similaire au modèle complet
matchin_images_treshold = 88
# les images ayant plus de X% de ressemblance seront supprimé valeur comprise entre 0 et 1
filtering_treshold = 0.80
#Toute les combiens de frames on check la vidéo faire sec*60
frame_gap = 300

"""if "__main__" == "__main__":
    # Load the OpenAI CLIP Model
    print('Loading CLIP Model...')
    #model = SentenceTransformer('clip-ViT-B-32')
    get_images(query,api_key,cx)
    #scan_video(VIDEO_SCANNED)"""


"""#test the time with cpu
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')
#get_images(query,api_key,cx,num_requests)
#filter_image()
execution_time = timeit.timeit(scan_video(VIDEO_SCANNED), number=1)
print(f"Execution time: {execution_time}")"""

# Initialize the DirectML device (this will use AMD GPU)
device = torch_directml.device()
# Load the model and move it to DirectML device (AMD GPU)
model = SentenceTransformer('clip-ViT-B-32').to(device)

#get_images(query,api_key,cx,num_requests)
#filter_image()
total_time = timeit.timeit(lambda : scan_video(VIDEO_SCANNED), number=1)
print(f"Mean time {total_time:.3e} s")