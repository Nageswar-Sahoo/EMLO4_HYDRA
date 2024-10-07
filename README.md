<h1>Dog Breed Image Dataset Training, Evaluation, and Inference with Docker </h1>

<h3>About Dataset</h3>

Description

This dataset contains a collection of images for 10 different dog breeds, meticulously gathered and organized to facilitate various computer vision tasks such as image classification and object detection. The dataset includes the following breeds:

       Golden Retriever
       German Shepherd
       Labrador Retriever
       Bulldog
       Beagle
       Poodle
       Rottweiler
       Yorkshire Terrier
       Boxer
       Dachshund

Each breed is represented by 100 images, stored in separate directories named after the respective breed. The images have been curated to ensure diversity and relevance, making this dataset a valuable resource for training and evaluating machine learning models in the field of computer vision.


Model 
    Model Architecture

Docker build

 1. Build the Docker image:

          docker build -t dogbreed-classification .
 
 3. To run training:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification train
 
 5. To run evaluation:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification eval
    
 7. To run inference:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer

 8. By default it performs inference on the images present in the input_images folder.

          To modify the infer arguments, you can do the following:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer --input_folder="path/to/custom/input" --  output_folder="path/to/custom/output" --     ckpt_path="path/to/custom/checkpoint.ckpt"

<h3>Prediction Results</h3>

  The model prediction gets saved in the predicted_images folder in the model artifacts.

  
    

