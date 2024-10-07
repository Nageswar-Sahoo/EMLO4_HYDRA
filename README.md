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


  <table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/92cb4dd8-8f37-4ae5-a906-7447554720a9" width="200"/></td>
    <td><img src="https://github.com/user-attachments/assets/2cde3ce2-3596-4bf9-bb88-9d2c996407eb" width="200"/></td>
    <td><img src="![image](https://github.com/user-attachments/assets/97d12ed1-9f40-4d9e-929f-d282be82e882)
" width="200"/></td>
  </tr>
  <tr>
    <td><img src="![image](https://github.com/user-attachments/assets/c2a4bc37-078d-49d2-afbe-65cc0ccd3a82)
" width="200"/></td>
    <td><img src="![image](https://github.com/user-attachments/assets/7ca0ca0e-3295-49eb-a266-e2009eca271e)
" width="200"/></td>
    <td><img src="![image](https://github.com/user-attachments/assets/3adf2fc0-2753-4cb9-bd93-b4bbcb236077)
" width="200"/></td>
  </tr>
</table>



  
    

