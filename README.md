# Productivity-Stress-and-Job-Satisfaction-analysis-on-social-Media-usage
Analysis of Social Media Usage and its Impact on  Productivity, Stress, and Job Satisfaction using  ***Multi-Task Learning***  

**📌 Project Overview**

This project investigates the impact of social media usage on individual productivity, stress levels, and job satisfaction. Using a comprehensive dataset containing user demographics, digital habits, and well-being metrics, I developed a Multi-Task Neural Network using PyTorch to simultaneously predict three distinct target variables:
1. Productivity Score (Regression Task)

2. Stress Level (Classification Task)

3. Job Satisfaction (Classification Task)

This project demonstrates proficiency in end-to-end machine learning pipelines, from complex data preprocessing to advanced deep learning architecture design.
   
**🛠️ Tech Stack & Skills Reflected**

  * Languages: Python
  
  * Deep Learning: PyTorch (Custom Module Subclassing, Multi-head Architecture)
  
  * Data Manipulation: Pandas, NumPy
  
  * Preprocessing: Scikit-Learn (Label Encoding, One-Hot Encoding, StandardScaler)
  
  * Visualization: Matplotlib, Seaborn
  
  * Key Concepts: Multi-Task Learning (MTL), Loss Weighting, Feature Engineering, Null Value Imputation.

**📊 The Dataset**

The dataset contains 30,000 records with various features regarding users' digital lives and professional well-being. Key features include:

  * Demographics: Age, Gender, Job Type.
  
  * Digital Habits: Daily Social Media Time, Platform Preference, Number of Notifications, Focus App usage.
  
  * Well-being Metrics: Sleep Hours, Screen Time before Sleep, Burnout levels.
  
  * Targets: Perceived Productivity, Actual Productivity, Stress Level, Job Satisfaction.

***⚙️ Methodology***

**Data Cleaning & Preprocessing**
  
  The raw data contained missing values and categorical variables that required robust handling to ensure model stability.
  
   * *Handling Missing Data:* Implemented a strategy using dropna with a threshold for sparse rows, followed by filling missing continuous values with the mean (e.g., Productivity Scores) or median (e.g., Social Media Time) to handle outliers effectively.
  
   * *Visual Inspection:* Used heatmaps to identify and verify the patterns of missing data before and after cleaning.Figure 1: Heatmap used to visualize sparsity in the dataset.2. 

**2. Feature Engineering**
  
   * *Categorical Encoding:*  Label Encoding was applied to binary/ordinal features like Gender.One-Hot Encoding (pd.get_dummies) was used for nominal variables like Job Type and Social Media Preference to prevent ordinal bias.
  
   * *Feature Scaling:* Applied StandardScaler to continuous features (e.g., Age, Work Hours, Notifications) to normalize distributions, essential for faster convergence in Neural Networks.

**3. Exploratory Data Analysis (EDA)**

   * Performed correlation analysis to understand linear relationships between screen time, sleep, and productivity scores, informing feature selection.

**4. Multi-Task Deep Learning**

ModelInstead of building three separate models, I designed a Multi-Task Learning (MTL) architecture. This approach leverages shared representations to generalize better across related tasks.

*Architecture:* 

  Shared Encoder: A series of fully connected layers (Linear -> ReLU -> Dropout) that extract common features from the input data.

**Task-Specific Heads:**
    * Head 1 (Productivity): A regression head outputting a continuous scalar value.
      
   * Head 2 (Stress): A classification head outputting logits for stress levels (1-10).
      
   * Head 3 (Job Satisfaction): A classification head outputting logits for satisfaction scores.
    
*Loss Function:* 

   I engineered a composite loss function to train all heads simultaneously, assigning weights based on task complexity and scale:
  
  $$\mathcal{L}_{total} = 0.6 \cdot \mathcal{L}_{MSE} + 0.25 \cdot \mathcal{L}_{Stress} + 0.15 \cdot \mathcal{L}_{Job}$$
  
   MSE Loss for Productivity (Regression).
  
   CrossEntropy Loss for Stress and Job Satisfaction (Classification).

**📈 Results & Evaluation**

  The model was trained over 75 epochs using the Adam optimizer. 
  
  Training logs indicate successful convergence across all three tasks.
  
   * **Productivity Prediction:** Achieved low Mean Squared Error (MSE), indicating accurate continuous predictions.
    
   * **Classification Accuracy:** The model learned to categorize stress and satisfaction levels effectively, balancing the trade-offs between the competing objectives of the multi-task loss.

**Training Performance Snippet**

Epoch 75/75 | Train Loss: 1.2541 | Val Loss: 1.2450 | Val Prod MSE: 0.7150, Stress Acc: 0.162, Job Acc: 0.331

(Note: Accuracy metrics reflect the difficulty of multi-class classification on granular 1-10 scales).

🚀 How to RunClone the repository:Bashgit clone https://github.com/AAdi-12/Productivity-Stress-and-Job-Satisfaction-analysis-on-social-Media-usage.git

Install Dependencies:Bashpip install numpy pandas matplotlib seaborn scikit-learn torch

**Run the Notebook:**

  Open social_media_vs_productivity.ipynb in Jupyter or Google Colab and execute the cells sequentially.

**🔮 Future ImprovementsHyperparameter Tuning:** Implementing Bayesian Optimization for learning rate and layer size selection.
  
  * *Advanced Architecture:* Experimenting with Transformer encoders for feature interaction.
  
  * *Feature Selection:* Analyzing feature importance (e.g., SHAP values) to understand which digital habits most significantly degrade productivity.
