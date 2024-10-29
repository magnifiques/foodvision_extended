![1728345339413](https://github.com/user-attachments/assets/799c1100-4058-44c4-ab8a-5c7dab1539d4)

# FoodVision Extended 🍔🥗🍕


Welcome to **FoodVision Extended**! This project builds upon the original **FoodVision Mini** to classify **20 different food items** using deep learning, specifically leveraging **Vision Transformers (ViT)**. From handling the Food101 dataset to deploying the model on Hugging Face, this project explores food classification on a larger, more complex scale.

Over the past few weeks, I’ve worked on an exciting computer vision project: a food classification model that recognizes 20 food items using deep learning. From handling the Food101 dataset to deploying the model on Hugging Face, I’ve gained valuable insights into building, training, and fine-tuning machine-learning models. In this article, I’ll walk you through the entire process, including the challenges I faced, like overfitting, and how I overcame them with data augmentation and advanced techniques like early stopping and scheduling.

## Overview
FoodVision Extended is a computer vision model that recognizes **20 food categories**. It builds on **FoodVision Mini** (which classified pizza, steak, and sushi) and explores advanced techniques in deep learning to improve accuracy and performance for a more extensive dataset.

## Looking Back: The Original FoodVision Mini

Before diving into my food classification model, I want to give a shout-out to the project that inspired me: **FoodVision Mini**. This project was initially developed to classify just three food categories—**pizza 🍕, steak 🥩, and sushi 🍣**—using the **Vision Transformer (ViT) B16** model.

The original FoodVision Mini was impressive for its simplicity and performance, focusing on demonstrating how Vision Transformers could be applied to image classification tasks. By leveraging ViT's attention mechanism, which breaks an image into small patches and treats each patch like a "word," the model achieved high accuracy while capturing both local and global patterns in the images.

## Project Goals

While FoodVision Mini was exciting, I aimed to take this idea a step further by:

- **Expanding the number of food categories**: My model now classifies **20 different food items** instead of just three.
- **Overcoming limitations of training time and resource usage**: By experimenting with various techniques like **data augmentation**, **schedulers**, and **early stopping**, I optimized the model for better performance over extended training periods.

This new project builds on the foundation laid by FoodVision Mini, introducing more complexity and a broader range of foods. This makes it more useful for real-world applications, such as **restaurant menu scanning** or **food/calorie-tracking apps**.

## Why the Upgrade?

The motivation behind expanding beyond the original FoodVision Mini was straightforward: to test the boundaries of what could be achieved with more diverse data. While the original project was an excellent starting point, scaling up the model to classify more food categories presented unique challenges:

- **Handling Visual Diversity**: More categories mean greater variance in textures, shapes, and colors.
- **Managing Overfitting and Generalization**: With more food items, the model needs to generalize well across different types of images—something I struggled with early on.
- **Efficient Training**: Training the model with 20 categories took longer, making it crucial to optimize the training process using techniques like **early stopping** and **learning rate scheduling**.
- **More Categories**: Instead of 3, the model now classifies 20 food items, making it more versatile for food recognition tasks.
- **Enhanced Model Training**: I implemented advanced techniques to combat overfitting and optimize training, like data augmentation, schedulers, and early stopping, which helped stabilize the model.
- **Better Deployment**: While FoodVision Mini was a great demo, deploying my version on Hugging Face Spaces ensures that anyone can interact with the model live, testing its ability to classify a wider range of food.
Through these upgrades, my goal was not just to recreate what FoodVision Mini had accomplished, but to enhance it, making the model more robust and scalable for real-world scenarios.

## Dataset: Food101 with 20 Classes

For this extended model, I worked with the Food101 dataset, a large collection of images featuring 101 different types of food. However, to keep my focus narrow and manageable, I selected 20 diverse food categories that would still provide ample variety for the training process while ensuring the training time remained reasonable. The Food101 dataset provided a solid foundation for training, with diverse and challenging examples across all 20 selected categories. This diversity pushed the model to capture fine details, making it a great learning experience in balancing data diversity and model performance.

![1728346573628](https://github.com/user-attachments/assets/a93f45c4-a988-4e31-a508-adcc0459840d)

## Dataset Details: Image Distribution

For this project, I organized my dataset into three distinct subsets to ensure a well-rounded evaluation of the model's performance:

- **Training Set**: 1,000 images – This set is used to train the model, allowing it to learn the features and characteristics of each food category.
- **Validation Set**: 500 images – This set is used during training to tune the model's hyperparameters and make adjustments. It helps assess how well the model generalizes to unseen data.
- **Test Set**: 150 images – This set is reserved for final evaluation after the model has been trained. It provides a clear measure of how the model performs on completely new images.

This structured approach to dataset distribution ensures that the model is adequately trained, validated, and tested, leading to a more reliable assessment of its performance in classifying the 20 food categories.

## Training the Model

I trained the model for 20 epochs, which took about 4 hours in total. During this training phase, the model improved its ability to recognize the food categories. However, I encountered some challenges, particularly with overfitting. This meant that while the model excelled at classifying training data, it struggled when presented with new, unseen images.

To address this issue, I utilized a Vision Transformer (ViT) architecture, which has shown remarkable performance in image classification tasks. ViT uses an attention mechanism that enables the model to focus on important features within images rather than relying solely on local patterns like traditional convolutional neural networks (CNNs). This capability allows ViT to capture global dependencies in the data, making it a suitable choice for the complexity of food images.

Despite the initial overfitting concerns, the model has made significant strides in recognizing and classifying the various food items effectively.

## Handling Overfitting: Data Augmentation and Early Stopping

To reduce overfitting and make the model generalize better, I used:

- **Data Augmentation**: Flipping, rotating, zooming, and shifting images created more diversity in the dataset, forcing the model to learn generalized patterns.
- **Learning Rate Scheduler**: This helped by gradually reducing the learning rate, ensuring the model didn’t make drastic updates to the weights in later stages of training.
- **Early Stopping**: By stopping the training process when the validation loss stopped improving, I avoided over-training the model.


## Performance Metrics:

Here’s how the model performed after 20 epochs of training:

- **Accuracy**: 86.32%
- **Precision**: 82.97%
- **Recall**: 80.32%
- **F1 Score**: 80.23%

Although the model isn’t perfect (about 86% accuracy overall), it performs well in predicting the majority of food items accurately and quickly! This is a significant improvement from my original model, and I plan to keep refining it to improve these metrics further.

![1728346217749](https://github.com/user-attachments/assets/86d6ea2b-a7d5-46fa-9878-ca8a258f36e0)

## How Can the Model Be Improved?

While the model is performing well, there are several ways to enhance it further:

- **Expand the Dataset**: The current dataset of **1,000 images per category** is limited. Adding more images or using **data augmentation** techniques (such as rotation, zoom, and flips) can introduce more diversity, helping the model generalize better.
  
- **Hyperparameter Tuning**: The training parameters could be optimized further. Using techniques like **grid search** or **learning rate scheduling** can help find the most efficient settings for faster and better learning.

## Deployment on Hugging Face

Once the model was trained, I deployed it on **Hugging Face** to make it accessible to everyone. Hugging Face Spaces offers a user-friendly interface where anyone can test the model in real time. You can try it out here: [Foodvision Extended](https://huggingface.co/spaces/vapit/Foodvision-Extended) 

## What's Next?

In the future, I plan to expand the model to classify even more food items, making it even more versatile and useful. Additionally, I will explore other model architectures beyond the Vision Transformer to see if they can improve accuracy and performance. This exploration could lead to discovering new techniques and strategies in food classification, ultimately enhancing the user experience and practical applications of the model.

## Conclusion

Building and deploying this food classification model was a rewarding experience that taught me the importance of overcoming common machine-learning challenges, such as overfitting and tuning hyperparameters. I learned valuable lessons in tackling these challenges for better performance.

If you’re working on similar projects, want to collaborate, or simply want to learn how I accomplished this, feel free to reach out! I’d love to connect and hear your thoughts!


## Live Demo
Test out the model live on Hugging Face Spaces: [FoodVision Extended](https://huggingface.co/spaces/vapit/Foodvision-Extended) 
