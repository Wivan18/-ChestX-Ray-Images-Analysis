# Deep Learning Analysis of X-Ray Images for Pneumonia Detection

## 1. Business Understanding

### a) Introduction

Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid, causing cough, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses, and fungi, can cause pneumonia. Pneumonia can range in seriousness from mild to life-threatening and it is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems ([Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204#:~:text=Pneumonia%20is%20an%20infection%20that,and%20fungi%2C%20can%20cause%20pneumonia.)).

The rapid advancement of Artificial Intelligence (AI) has brought significant benefits across various industries, including healthcare. Traditionally, diagnosing pneumonia requires time-consuming physical examinations and lab tests, often necessitating multiple doctor visits. To address this issue, we aim to develop a deep learning model capable of accurately detecting pneumonia from chest x-ray images. Such a tool holds immense value for healthcare professionals and patients, enabling quicker and more precise diagnoses. Radiologists and other specialists can leverage this technology to enhance their diagnostic accuracy, ultimately leading to better patient care and treatment outcomes.

### b) Problem Statement

**What is the prevailing circumstance?**

Early detection and treatment of pneumonia are essential for avoiding complications and enhancing clinical results. Detecting pneumonia is not only a medical necessity but also a humanitarian imperative and a technological frontier. Chest X-rays are a frequently used imaging modality for diagnosing pneumonia.

**What problem is being addressed?**

According to the [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/pneumonia), Pneumonia accounts for 14% of all deaths of children under 5 years old, killing 740,180 children in 2019. Chest X-ray imaging serves as a prevalent diagnostic tool for pneumonia, offering insights such as increased lung opacity. Nevertheless, interpreting chest X-rays poses challenges due to the subtle nature of pneumonia symptoms and their potential overlap with other respiratory conditions. Analyzing radiological images, including chest X-rays and CT scans, demands specialized expertise and can consume significant time in the diagnostic process.

**How the project aims to solve the problem?**

To address this issue, this project aims to develop a deep learning model capable of accurately detecting pneumonia from chest x-ray images. Such a tool holds immense value for healthcare professionals and patients, enabling quicker and more precise diagnoses. Radiologists and other specialists can leverage this technology to enhance their diagnostic accuracy, ultimately leading to better patient care and treatment outcomes.

### c) Objectives

#### Main Objectives

- Develop a deep learning model to accurately identify pneumonia from chest x-ray images.

#### Specific Objectives

- Preprocess the chest X-ray images to standardize resolution, orientation, and contrast, optimizing them for input into the deep learning model.
- Explore and implement various deep learning architectures suitable for image classification tasks, such as convolutional neural networks (CNNs), to identify the most effective model for pneumonia detection.
- Train the selected deep learning model using the prepared dataset, employing techniques like data augmentation to enhance model generalization and prevent overfitting.
- Evaluate the trained model's performance using appropriate metrics such as accuracy, sensitivity, specificity, and area under the receiver operating characteristic (ROC) curve, validating its effectiveness in pneumonia detection.
- Fine-tune the model parameters and architecture based on evaluation results, iteratively improving its performance and robustness.

### d) Notebook Structure

i) Business Understanding<br>
ii) Data Understanding<br>
iii) Exploratory Data Analysis<br>
iv) Data Preprocessing<br>
v) Modeling<br>
vi) Evaluation<br>
vii) Conclusion<br>
viii) Recommendation<br>
ix) Next Steps<br>

### e) Stakeholders

Key stakeholders interested in leveraging deep learning for medical imaging include: healthcare professionals, patients, hospitals, medical device manufacturers, and insurance companies. For instance, hospitals can optimize resource allocation and improve treatment efficacy, while medical device manufacturers can enhance product development for more accurate diagnoses. Additionally, researchers and government agencies stand to benefit from these advancements, using the models to deepen disease understanding and ensure regulatory compliance.

### f) Metric of Success

The performance of the model is evaluated based on:
* Accuracy - achieving an accuracy of over 75%
* Loss Function - minimize loss function to ensure the model is not overfitting

## 2. Data Understanding

The dataset used in this project was obtained from [Mendley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3). It comprises 5,863 JPEG images categorized into two classes: "Pneumonia" and "Normal." It is organized into three main folders: "train," "test," and "val," each containing subfolders corresponding to the image categories.

The chest X-ray images, captured in the anterior-posterior view, were obtained from pediatric patients aged one to five years old at Guangzhou Women and Children’s Medical Center, Guangzhou. These images were part of routine clinical care procedures.

Before inclusion in the dataset, all chest radiographs underwent a quality control process to remove any low-quality or unreadable scans. Subsequently, the diagnoses assigned to the images were graded by two expert physicians. To mitigate potential grading errors, a third expert also evaluated the images in the evaluation set.

Overall, this dataset provides a curated collection of chest X-ray images, ensuring quality and accuracy through rigorous quality control measures and expert evaluation, making it suitable for training AI systems for pneumonia diagnosis.

## 3. Exploratory Data Analysis

This step involves examining and understanding the dataset before applying machine learning algorithms. It will guide in processes like: data preprocessing, model selection, and performance evaluation strategies.

The EDA performed on the dataset includes visualizing samples of normal and pneumonia chest X-ray images, and plotting the class distributions for the training, test, and validation sets.

## 4. Data Preprocessing

### Extracting the images and labels

The images and labels were extracted from the train, test, and validation generators, and returned as arrays. The train_images array holds the image data, while the train_labels array contains corresponding labels.

### Checking the information of the datasets

Information regarding the shape and size of both the image and label datasets was explored and displayed. Printing these values provides an overview of the dataset's sample count and the dimensions of the image and label arrays.

### Reshaping the images

The images were converted from their original 3-dimensional shape (height x width x channels) into a flattened format. This flattened representation allows the images to be easily processed and fed into machine learning models.

## 5. Modeling

### Baseline model: A Densely Connected Neural Network

Initially, a baseline fully connected network is constructed utilizing the Keras Sequential API. This network consists of two hidden layers followed by an output layer. The first two layers incorporate ReLU activation functions, which introduce non-linearity to the network, enabling it to learn complex patterns in the data. Meanwhile, the final layer employs a sigmoid activation function, which outputs probabilities between 0 and 1, making it suitable for binary classification tasks.

### Model 2: Convolutional Neural Network

The second model comprises a sequential architecture consisting of two convolutional layers, each followed by max pooling layers. These convolutional layers are responsible for extracting features from the input images, while the subsequent max pooling layers reduce the spatial dimensions of the feature maps. Finally, the model concludes with a dense layer, which performs the classification task.

### Model 3: CNN with Architecture modifications

The architecture of the model was modified by adding more convolutional layers, increasing the number of filters in each layer, and introducing two additional dense layers after the flattening layer.

## 6. Model Evaluation

The performance of the two tuned models (Model 2 and Model 3) is compared using the following metrics:
* Test Loss
* Test Accuracy
* Test Precision

## 7. Conclusion

The developed deep learning models, particularly Model 3, have shown promising results in accurately detecting pneumonia from chest X-ray images. The models were able to achieve high training accuracy and reasonable validation/test performance, demonstrating their potential for real-world deployment in clinical settings.

The best performing model, "model_3", achieved a test Accuracy of 75% indicates that the model is able to correctly classify 75% of the test samples. This test accuracy although not bad could be improved by using more data such as larger and more diverse dataset of chest X-ray images.

## 8. Recommendation

Based on the evaluation results, we recommend the deployment of Model 3 as the primary model for pneumonia diagnosis from chest X-rays. This model has exhibited the best overall performance, striking a balance between high accuracy, precision, and low loss. Its integration into hospital workflows and medical imaging analysis systems can significantly enhance the speed and accuracy of pneumonia diagnosis, ultimately leading to improved patient outcomes and more efficient healthcare processes.

Prospective Clinical Validation: Collaborate with healthcare providers to deploy the model in a real-world clinical setting and conduct a comprehensive evaluation of its performance, practicality, and impact on patient care.

## 9. Next Steps

To further improve the model's performance and ensure its robustness, we suggest the following future work:
* **Expand the Dataset:** Acquire a larger and more diverse dataset of chest X-ray images to improve the model's generalization capabilities and its ability to handle a wider range of pneumonia cases.
* **Pneumonia Severity Classification:** Instead of just a binary classification (pneumonia vs. normal), extend the model to classify the severity of pneumonia (mild, moderate, severe). This would provide even more actionable insights for doctors.
* **Pneumonia Type Classification:** Differentiate between bacterial and viral pneumonia. This information can guide treatment decisions as antibiotics are ineffective against viruses.
* **Localization of Pneumonia:** Pinpoint the specific regions of the lung affected by pneumonia in the X-ray image. This can be crucial for further investigation and treatment planning.
* **Cloud-Based Deployment:** Deploy the model on a cloud platform for remote access and scalability. This would allow for wider adoption and utilization from various hospitals and clinics.
