import torch
from PIL import Image
import clip
import os
import csv

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("best.pt", device=device)
    model, preprocess = clip.load('ViT-B/32', device='cuda')

    imagePath_and_text = []

    urn = 6640106
    files = os.listdir(f"./{urn}")
    group_count = len(files)

    for i in range(group_count):

        image_path = f"./{urn}/group{i}/images"
        print(image_path)
        images = [preprocess(Image.open(f"{image_path}/{img_name}")).unsqueeze(0).to(device) for img_name in ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg"]] 
        # images = [preprocess(Image.open("./6640106/group0/images/1.jpg")).unsqueeze(0).to(device)]

        # captions = ["Two black dogs are playing tug-of-war with an orange toy .", "Two dogs playing with a ball .", "a man in a wetsuit is surfing", "Two black dogs are bearing their teeth beside a white couch .",  "Two girls swing on with a boy , all three are wearing blue shirts of the same shade and blue jeans .", "A girl dressed in a red polka dot dress holds an adults hand while out walking .", "A man is playing the guitar for a child in a hospital bed", "A person hanging from a rocky cliff .", "A man is a black ninja suit with a mask is playing a guitar .", "Two large black dogs are playing in a grassy field ."]

        captions = []
        with open(f'{urn}/group{i}/captions_candidates.txt', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Join the row elements with commas and add to the list
                captions.append(','.join(row))

        # Preprocess the text
        text = clip.tokenize(captions).to(device)

        # Compute the image and text features
        with torch.no_grad():
            image_features = [model.encode_image(image) for image in images]
            text_features = model.encode_text(text)

        # Compute the similarity between the image and text features and select the best captions
        png_counter = 0
        for image_feature in image_features:
            similarities = (image_feature @ text_features.T).softmax(dim=-1)
            best_caption_index = similarities.argmax().item()

            imagePath_and_text.append((f"group{i}/images/{png_counter}.jpg", captions[best_caption_index]))

            print(f"{captions[best_caption_index]} and the image {png_counter}.jpg")
            png_counter += 1

        with open('file.csv', 'w', newline='') as f:
            writer = csv.writer(f)

            # Write each row to the CSV file
            for row in imagePath_and_text:
                writer.writerow(row)