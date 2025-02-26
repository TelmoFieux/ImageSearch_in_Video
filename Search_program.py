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

    #on met a la poubelle les image transparentes ou avec un ratio non voulu
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
        
#Cette fonction calcule la similarité entre toute les pairs le nombre de comparaison est donc exponentielle

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

def find_matching_imagesV2(frame_path,encoded_images):
    # Encode the image you want to compare, and move it to DirectML device
    frame_image_embedding = model.encode([Image.open(frame_path)], batch_size=1 ,convert_to_tensor=True, device=device)

    # Compute cosine similarities between the frame image and the other images (on AMD GPU)
    similarities = util.cos_sim(frame_image_embedding, encoded_images)

    # Go through the similarity scores
    max=0
    for i, score in enumerate(similarities[0]):
        score_percentage = score.item() * 100  # Convert the similarity score to percentage
        if score_percentage > max:
            max = score_percentage

    # Return 0 if no similar image meets the threshold
    return max

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

def new_frame_gap(prob,prev_prob,nb_frame,current_gap):
    for k in seuil.keys():
            if prob<k:
                gap=seuil[k]
                new_frame=nb_frame
                nb_etape=0
                if file_de_voiture:
                    if prob!=0 and prob>prev_prob and prev_prob < k-5:
                        while(new_frame-(gap*60)>nb_frame-(current_gap*60)):
                            new_frame=new_frame-(gap*60)
                            nb_etape+=1
                        print("Deceleration, on recheck a partir de la frame " +str(new_frame) +" alors qu'on était a la frame" + str(nb_frame) + "On est revenu en arrière de :" + str(nb_etape) + "\n")
                break
    return gap,new_frame

def scan_video(videopath):
    # Open the video file
    video = cv2.VideoCapture(videopath)
    # Initialize variables
    found_timestamps = {}
    video_lenght = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    encode_images = encode_les_images()
    current_frame=0
    prev_prob=0
    frame_gap=60 #Can be changed

    # Loop through each frame of the video
    while (current_frame < video_lenght):

        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()

        if not ret:
            break
        print('minute elapsed', current_frame / 60)
        
        #au cas ou il y a unne erreur au moment de la réécriture on relance le processus
        if cv2.imwrite('extracted_frame.jpg', frame) == False:
            cv2.imwrite('extracted_frame.jpg', frame)

        # Perform template matching
        prob = find_matching_imagesV2('.\extracted_frame.jpg',encode_images)
        print(prob)

        # If match is above threshold, record the timestamp
        if prob >= matchin_images_treshold:
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds
            found_timestamps[timestamp/60]=prob
        
        frame_gap,current_frame=new_frame_gap(prob,prev_prob,current_frame,frame_gap)
        prev_prob = prob
        current_frame+=frame_gap*60

    video.release()
    os.remove("extracted_frame.jpg")

    # Output the found timestamps
    print("image found at timestamps:", found_timestamps)


#? KEY PARAMETERS

VIDEO_SCANNED = "Ratchet_et_clank_2_part_7_CRF.mp4"
# Your API key for the google search if you want to use it
api_key = "your api key"
# Your Custom Search Engine ID for the google image search
cx = "your id"
# The search query if you use google image search
query = "Your search query"
# Number of request (10 images per request) for the google image search
num_requests = 2
# seuil d'acceptation pour la similarité entre image valeur comprise entre 0 et 100
# Si une frame de la vidéo est au dessus de X% on considère que c'est un succès et qu'on a trouvé ce que l'on cherchait
matchin_images_treshold = 88
# les images ayant plus de X% de ressemblance parmis les images obtenu après la recherche via google image search seront supprimé valeur comprise entre 0 et 1
filtering_treshold = 0.80
#Seuil pour la vitesse adaptative [seuil en pourcentage : vitesse en seconde] utilisé si file_de_voiture = True. Explication de cette méthode dans le fichier ReadMe
seuil={75:20,80:10,85:5,90:2,100:2}
#Décide si on utilise la méthode file de voiture expliquer dans performane.txt| Permet une approche intélligente qui mixe rapidité et fiabilité.
file_de_voiture=True

# Initialize the DirectML device
device = torch_directml.device()
# Load the model and move it to DirectML device
model = SentenceTransformer('clip-ViT-B-32').to(device)
#get_images(query,api_key,cx,num_requests) if you use the google image search
#filter_image()
scan_video(VIDEO_SCANNED)