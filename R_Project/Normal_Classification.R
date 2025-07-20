library(tidyverse)
library(caret)
library(pROC)
library(MLmetrics)
library(randomForest)
library(e1071)
library(VIM)          # For missing data visualization
library(ggcorrplot)   # Nice correlation heatmaps
library(gridExtra)    # Arrange plots
library(reshape2)     # For reshaping data for barplots
library(cowplot)    
library(corrplot)   # For ggplot arrangement
library(pheatmap) 

# --- Load and preprocess data ---

data <- read.csv("C:/Users/slimc/Desktop/R_Project/alzheimers_disease_data.csv", stringsAsFactors = TRUE)
data$Diagnosis <- as.factor(data$Diagnosis)
levels(data$Diagnosis) <- make.names(levels(data$Diagnosis))
if ("ID" %in% names(data)) data <- data %>% select(-ID)



# --- PREPROCESSING DIAGNOSTICS ---

# 1) Missing values visualization
png("missing_values_before.png", width=800, height=500)
aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE,
     labels=names(data), cex.axis=.7, gap=3, ylab=c("Missing data","Pattern"))
dev.off()

# 2) Remove constant columns
constant_cols <- sapply(data, function(x) length(unique(x)) == 1)
cat("Constant columns removed:\n")
print(names(data)[constant_cols])
data <- data %>% select(-which(constant_cols))


# 3) Impute missing values (numeric: median, factor: mode)
num_cols <- sapply(data, is.numeric)
for (col in names(data)[num_cols]) {
  median_val <- median(data[[col]], na.rm = TRUE)
  data[[col]][is.na(data[[col]])] <- median_val
}
factor_cols <- sapply(data, is.factor)
for (col in names(data)[factor_cols]) {
  mode_val <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
  data[[col]][is.na(data[[col]])] <- mode_val
}
num_data <- data %>% select(where(is.numeric))
cor_matrix <- cor(num_data, use = "pairwise.complete.obs")

p_corr <- ggcorrplot(cor_matrix, 
                     hc.order = TRUE,        # reorder variables by hierarchical clustering
                     type = "lower",         # only lower triangle
                     lab = TRUE,             # add correlation coefficients
                     lab_size = 2, 
                     title = "Correlation Matrix of Numeric Variables",
                     colors = c("blue", "white", "red"))
ggsave("correlation_heatmap.png", p_corr, width = 8, height = 6)   # <== Plot sauvegardé ici

num_data <- data %>% select(where(is.numeric))
cor_matrix <- cor(num_data, use = "pairwise.complete.obs")

# Corrplot avec clustering hiérarchique (order = "hclust")
png("correlation_dendro.png", width=800, height=800)
corrplot(cor_matrix, method = "color", order = "hclust", 
         addrect = 4,  # dessine 4 rectangles pour clusters
         tl.cex = 0.7, tl.col = "black",
         cl.cex = 0.8,
         title = "Matrice de corrélation avec regroupement hiérarchique",
         mar=c(0,0,2,0))
dev.off()  # <== Plot sauvegardé ici

# 4) Remove highly correlated numeric features (>0.8)
num_data <- data %>% select(where(is.numeric))
cor_matrix <- cor(num_data, use = "pairwise.complete.obs")
high_cor_features <- findCorrelation(cor_matrix, cutoff = 0.8, names = TRUE)
cat("Highly correlated features to consider removing:\n")
print(high_cor_features)
data_reduced <- data %>% select(-all_of(high_cor_features))

# Correlation heatmaps before and after removing highly correlated features
corr_all <- cor(num_data, use = "pairwise.complete.obs")
num_data_reduced <- data_reduced %>% select(where(is.numeric))
corr_reduced <- cor(num_data_reduced, use = "pairwise.complete.obs")

p_corr_before <- ggcorrplot(corr_all, hc.order = TRUE, type = "lower", lab = TRUE, lab_size=2,
                            title = "Correlation Matrix Before Removing High Correlations",
                            colors = c("blue", "white", "red"))
p_corr_after <- ggcorrplot(corr_reduced, hc.order = TRUE, type = "lower", lab = TRUE, lab_size=2,
                           title = "Correlation Matrix After Removing High Correlations",
                           colors = c("blue", "white", "red"))
png("correlation_before_after.png", width=1200, height=500)
grid.arrange(p_corr_before, p_corr_after, ncol=2)
dev.off()  # <== Plot sauvegardé ici

# 5) Class distribution plot
class_dist <- data %>% count(Diagnosis)
p_class_dist <- ggplot(class_dist, aes(x = Diagnosis, y = n, fill = Diagnosis)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Distribution of Diagnosis Classes", y = "Count", x = "Diagnosis") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position = "none")
ggsave("class_distribution.png", p_class_dist, width=6, height=4)   # <== Plot sauvegardé ici

# 6) Train-test split
set.seed(123)
train_idx <- createDataPartition(data_reduced$Diagnosis, p = 0.8, list = FALSE)
train_data <- data_reduced[train_idx, ]
test_data <- data_reduced[-train_idx, ]

# 7) Scale numeric features for clustering visualization
scaled_train_numeric <- scale(train_data %>% select(where(is.numeric)))
scaled_df <- as.data.frame(scaled_train_numeric)
scaled_df_long <- scaled_df %>% pivot_longer(cols = everything(), names_to = "Variable", values_to = "ScaledValue")
p_scaled_dist <- ggplot(scaled_df_long, aes(x = ScaledValue)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Distribution of Scaled Numeric Variables (Train Data)") +
  theme_minimal()
ggsave("scaled_variables_distribution.png", p_scaled_dist, width=10, height=6)  # <== Plot sauvegardé ici

# --- MODELING SETUP ---

ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
model_formula <- Diagnosis ~ .

# -----------------------
# MODELS WITHOUT CLUSTER
# -----------------------

set.seed(123)
model_knn <- caret::train(model_formula, data = train_data, method = "knn", trControl = ctrl, metric = "ROC",
                          preProcess = c("center", "scale"), tuneLength = 5)
pred_probs_knn <- predict(model_knn, newdata = test_data, type = "prob")[, levels(data$Diagnosis)[2]]
pred_class_knn <- predict(model_knn, newdata = test_data)
conf_mat_knn <- confusionMatrix(pred_class_knn, test_data$Diagnosis)
auc_knn <- auc(roc(test_data$Diagnosis, pred_probs_knn))
f1_knn <- F1_Score(y_true = test_data$Diagnosis, y_pred = pred_class_knn, positive = levels(data$Diagnosis)[2])

model_lr <- glm(model_formula, data = train_data, family = binomial)
pred_probs_lr <- predict(model_lr, newdata = test_data, type = "response")
pred_class_lr <- factor(ifelse(pred_probs_lr > 0.5, levels(data$Diagnosis)[2], levels(data$Diagnosis)[1]),
                        levels = levels(data$Diagnosis))
conf_mat_lr <- confusionMatrix(pred_class_lr, test_data$Diagnosis)
auc_lr <- auc(roc(test_data$Diagnosis, pred_probs_lr))
f1_lr <- F1_Score(y_true = test_data$Diagnosis, y_pred = pred_class_lr, positive = levels(data$Diagnosis)[2])

set.seed(123)
model_rf <- randomForest(model_formula, data = train_data)
pred_probs_rf <- predict(model_rf, newdata = test_data, type = "prob")[, 2]
pred_class_rf <- predict(model_rf, newdata = test_data, type = "response")
conf_mat_rf <- confusionMatrix(pred_class_rf, test_data$Diagnosis)
auc_rf <- auc(roc(test_data$Diagnosis, pred_probs_rf))
f1_rf <- F1_Score(y_true = test_data$Diagnosis, y_pred = pred_class_rf, positive = levels(data$Diagnosis)[2])

set.seed(123)
model_svm <- svm(model_formula, data = train_data, probability = TRUE)
svm_pred <- predict(model_svm, newdata = test_data, probability = TRUE)
pred_probs_svm <- attr(svm_pred, "probabilities")[, levels(data$Diagnosis)[2]]
pred_class_svm <- svm_pred
conf_mat_svm <- confusionMatrix(pred_class_svm, test_data$Diagnosis)
auc_svm <- auc(roc(test_data$Diagnosis, pred_probs_svm))
f1_svm <- F1_Score(y_true = test_data$Diagnosis, y_pred = pred_class_svm, positive = levels(data$Diagnosis)[2])

# ROC curves without clustering
roc_knn <- roc(test_data$Diagnosis, pred_probs_knn)
roc_lr <- roc(test_data$Diagnosis, pred_probs_lr)
roc_rf <- roc(test_data$Diagnosis, pred_probs_rf)
roc_svm <- roc(test_data$Diagnosis, pred_probs_svm)

p_roc_no_clust <- ggroc(list(KNN = roc_knn, LogisticRegression = roc_lr, RandomForest = roc_rf, SVM = roc_svm)) +
  labs(title = "ROC Curves - Models Without Clustering") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
ggsave("roc_curves_no_clustering.png", p_roc_no_clust, width=7, height=5)  # <== Plot sauvegardé ici

# -----------------------
# CLUSTERING
# -----------------------

clust_train <- train_data %>% select(where(is.numeric))
nzv <- nearZeroVar(clust_train)
if(length(nzv) > 0) clust_train <- clust_train[, -nzv]
clust_scaled <- scale(clust_train)
k <- 3
set.seed(123)
kmeans_res <- kmeans(clust_scaled, centers = k, nstart = 25)
train_data$Cluster <- factor(kmeans_res$cluster)

clust_test <- test_data %>% select(colnames(clust_train))
clust_test_scaled <- scale(clust_test,
                           center = attr(clust_scaled, "scaled:center"),
                           scale = attr(clust_scaled, "scaled:scale"))
assign_cluster <- function(newdata, centers) {
  # Convert to matrix if vector
  if (is.vector(newdata)) newdata <- matrix(newdata, nrow=1)
  
  # Ensure all data is numeric
  newdata <- matrix(as.numeric(newdata), nrow=nrow(newdata))
  centers <- matrix(as.numeric(centers), nrow=nrow(centers))
  
  # Calculate distances for each row
  result <- apply(newdata, 1, function(row) {
    dists <- apply(centers, 1, function(center) {
      sqrt(sum((row - center)^2))
    })
    which.min(dists)
  })
  
  # Return single value if input was single row, otherwise return vector
  if (nrow(newdata) == 1) result[1] else result
}
test_data$Cluster <- factor(assign_cluster(clust_test_scaled, kmeans_res$centers))

# PCA plot of clusters
pca <- prcomp(clust_scaled)
pca_df <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], Cluster = train_data$Cluster)
p_cluster <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(title = "Cluster Visualization (Train set, PCA components 1 & 2)") +
  theme_minimal() +
  scale_color_brewer(palette = "Dark2")
ggsave("clusters_pca.png", p_cluster, width=7, height=5)  # <== Plot sauvegardé ici

# Filter clusters with less than 2 members
remove_singleton_clusters <- function(train_df, test_df, cluster_col) {
  cluster_counts <- table(train_df[[cluster_col]])
  valid_clusters <- names(cluster_counts[cluster_counts >= 2])
  train_filtered <- train_df %>% filter(!!sym(cluster_col) %in% valid_clusters) %>% mutate(!!cluster_col := droplevels(!!sym(cluster_col)))
  test_filtered <- test_df %>% filter(!!sym(cluster_col) %in% valid_clusters) %>% mutate(!!cluster_col := factor(!!sym(cluster_col), levels = levels(train_filtered[[cluster_col]])))
  list(train = train_filtered, test = test_filtered)
}
filtered <- remove_singleton_clusters(train_data, test_data, "Cluster")
train_sub <- filtered$train
test_sub <- filtered$test

formula_cluster <- Diagnosis ~ . + Cluster

# -----------------------
# MODELS WITH CLUSTER
# -----------------------

set.seed(123)
model_knn_clust <- caret::train(formula_cluster, data = train_sub, method = "knn", trControl = ctrl, metric = "ROC",
                                preProcess = c("center", "scale"), tuneLength = 5)
pred_probs_knn_clust <- predict(model_knn_clust, newdata = test_sub, type = "prob")[, levels(data$Diagnosis)[2]]
pred_class_knn_clust <- predict(model_knn_clust, newdata = test_sub)
conf_mat_knn_clust <- confusionMatrix(pred_class_knn_clust, test_sub$Diagnosis)
auc_knn_clust <- auc(roc(test_sub$Diagnosis, pred_probs_knn_clust))
f1_knn_clust <- F1_Score(y_true = test_sub$Diagnosis, y_pred = pred_class_knn_clust, positive = levels(data$Diagnosis)[2])

model_lr_clust <- glm(formula_cluster, data = train_sub, family = binomial)
pred_probs_lr_clust <- predict(model_lr_clust, newdata = test_sub, type = "response")
pred_class_lr_clust <- factor(ifelse(pred_probs_lr_clust > 0.5, levels(data$Diagnosis)[2], levels(data$Diagnosis)[1]), levels = levels(data$Diagnosis))
conf_mat_lr_clust <- confusionMatrix(pred_class_lr_clust, test_sub$Diagnosis)
auc_lr_clust <- auc(roc(test_sub$Diagnosis, pred_probs_lr_clust))
f1_lr_clust <- F1_Score(y_true = test_sub$Diagnosis, y_pred = pred_class_lr_clust, positive = levels(data$Diagnosis)[2])

set.seed(123)
model_rf_clust <- randomForest(formula_cluster, data = train_sub)
pred_probs_rf_clust <- predict(model_rf_clust, newdata = test_sub, type = "prob")[, 2]
pred_class_rf_clust <- predict(model_rf_clust, newdata = test_sub, type = "response")
conf_mat_rf_clust <- confusionMatrix(pred_class_rf_clust, test_sub$Diagnosis)

# Matrice de confusion et visualisation pour Random Forest avec clustering
mat_rf_clust <- conf_mat_rf_clust$table

# Affichage console (déjà fait avec print(conf_mat_rf_clust), à garder)
print(conf_mat_rf_clust)

# Visualisation heatmap pheatmap
pheatmap(mat_rf_clust,
         display_numbers = TRUE,
         color = colorRampPalette(c("white", "grey"))(50),
         main = "Matrice de confusion - Random Forest avec clustering")

# Sauvegarde en PNG pour inclusion dans rapport LaTeX
png("confusion_matrix_rf_clust.png", width = 600, height = 500)
pheatmap(mat_rf_clust,
         display_numbers = TRUE,
         color = colorRampPalette(c("white", "grey"))(50),
         main = "Matrice de confusion - Random Forest avec clustering")
dev.off()  # <- impératif pour finaliser et fermer le fichier PNG


auc_rf_clust <- auc(roc(test_sub$Diagnosis, pred_probs_rf_clust))
f1_rf_clust <- F1_Score(y_true = test_sub$Diagnosis, y_pred = pred_class_rf_clust, positive = levels(data$Diagnosis)[2])

set.seed(123)
model_svm_clust <- svm(formula_cluster, data = train_sub, probability = TRUE)
svm_pred_clust <- predict(model_svm_clust, newdata = test_sub, probability = TRUE)
pred_probs_svm_clust <- attr(svm_pred_clust, "probabilities")[, levels(data$Diagnosis)[2]]
pred_class_svm_clust <- svm_pred_clust
conf_mat_svm_clust <- confusionMatrix(pred_class_svm_clust, test_sub$Diagnosis)
auc_svm_clust <- auc(roc(test_sub$Diagnosis, pred_probs_svm_clust))
f1_svm_clust <- F1_Score(y_true = test_sub$Diagnosis, y_pred = pred_class_svm_clust, positive = levels(data$Diagnosis)[2])

saveRDS(model_rf_clust, "model_rf_clust.rds")
saveRDS(kmeans_res$centers, "kmeans_centers.rds")

# ROC curves with clustering
roc_knn_clust <- roc(test_sub$Diagnosis, pred_probs_knn_clust)
roc_lr_clust <- roc(test_sub$Diagnosis, pred_probs_lr_clust)
roc_rf_clust <- roc(test_sub$Diagnosis, pred_probs_rf_clust)
roc_svm_clust <- roc(test_sub$Diagnosis, pred_probs_svm_clust)

p_roc_clust <- ggroc(list(KNN = roc_knn_clust, LogisticRegression = roc_lr_clust, RandomForest = roc_rf_clust, SVM = roc_svm_clust)) +
  labs(title = "ROC Curves - Models With Clustering") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
ggsave("roc_curves_with_clustering.png", p_roc_clust, width=7, height=5)  # <== Plot sauvegardé ici

# --- FINAL COMPARISON TABLE ---

comparison <- tibble(
  Model = rep(c("KNN", "Logistic Regression", "Random Forest", "SVM"), each = 2),
  Clustering = rep(c("Without Cluster", "With Cluster"), times = 4),
  Accuracy = c(conf_mat_knn$overall["Accuracy"], conf_mat_knn_clust$overall["Accuracy"],
               conf_mat_lr$overall["Accuracy"], conf_mat_lr_clust$overall["Accuracy"],
               conf_mat_rf$overall["Accuracy"], conf_mat_rf_clust$overall["Accuracy"],
               conf_mat_svm$overall["Accuracy"], conf_mat_svm_clust$overall["Accuracy"]),
  AUC = c(auc_knn, auc_knn_clust,
          auc_lr, auc_lr_clust,
          auc_rf, auc_rf_clust,
          auc_svm, auc_svm_clust),
  F1_Score = c(f1_knn, f1_knn_clust,
               f1_lr, f1_lr_clust,
               f1_rf, f1_rf_clust,
               f1_svm, f1_svm_clust)
)

print(comparison)

# Barplot for final report
comparison_melt <- melt(comparison, id.vars = c("Model", "Clustering"))
p_perf <- ggplot(comparison_melt, aes(x = Model, y = value, fill = Clustering)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free_y") +
  labs(title = "Model Performance Comparison With and Without Clustering",
       y = "Metric Value", x = "Model") +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired")
ggsave("performance_comparison.png", p_perf, width=9, height=5)  # <== Plot sauvegardé ici








##################################################### SHINY APP #####################################################



library(shiny)
library(reticulate)
library(dplyr)
library(randomForest)
library(bslib) # For modern Bootstrap themes

# Charger la fonction python d'extraction (adapter le chemin à ton fichier)
source_python("C:/Users/slimc/Desktop/R_Project/extract_info.py")

# Charger le modèle et centres kmeans
model_rf_clust <- readRDS("model_rf_clust.rds")
kmeans_centers <- readRDS("kmeans_centers.rds")

# Custom theme
memora_theme <- bs_theme(
  version = 5,
  bootswatch = "cosmo",
  primary = "#1a237e", # Deep blue
  secondary = "#1976d2", # Lighter blue
  success = "#43a047",
  info = "#0288d1",
  warning = "#fbc02d",
  danger = "#e53935",
  base_font = font_google("Roboto")
)

ui <- fluidPage(
  theme = memora_theme,
  tags$head(
    tags$style(HTML('
      .memora-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 120px;
        margin-bottom: 10px;
      }
      .memora-title {
        text-align: center;
        color: #1a237e;
        font-family: "Roboto", sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
      }
      .memora-panel {
        background: #f5f7fa;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(26,35,126,0.08);
        padding: 2rem 2rem 1.5rem 2rem;
        margin-bottom: 2rem;
      }
      .memora-btn {
        background-color: #1976d2 !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
      }
      .memora-output {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a237e;
        background: #e3eafc;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
      }
      .memora-label {
        font-weight: 500;
        color: #1976d2;
      }
      textarea {
        font-family: "Roboto Mono", monospace;
        font-size: 1rem;
      }
    ')),
    # Add Google Fonts
    tags$link(rel = "stylesheet", href = "https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap"),
    tags$link(rel = "stylesheet", href = "https://fonts.googleapis.com/css?family=Roboto+Mono:400,700&display=swap")
  ),
  # Logo and title
  tags$img(src = "logo.png", class = "memora-logo"),
  div(class = "memora-title", "Prédiction Alzheimer avec Memora"),
  br(),
  fluidRow(
    column(5,
      div(class = "memora-panel",
        h4(class = "memora-label", "Entrez toutes les variables dans ce format (exemples) :"),
        tags$div(style = "font-size:1rem; color:#333; margin-bottom:0.5rem;",
          "Age: 72", br(),
          "Gender: 1", br(),
          "BMI: 23.5", br(),
          "Smoking: 0", br(),
          "..."
        ),
        tags$textarea(id = "text_input", rows = 15, placeholder = "Entrez toutes les variables ici, chacune sur une ligne...", style = "width:100%; margin-bottom:1rem;"),
        actionButton("predict", "Prédire", class = "memora-btn")
      )
    ),
    column(7,
      div(class = "memora-panel",
        div(class = "memora-output", textOutput("main_output"))
      )
    )
  )
)

server <- function(input, output, session) {
  main_message <- reactiveVal("")

  # Load the model and get training data structure at startup
  model_data <- reactive({
    debug_print("Loading model from file", "model_rf_clust.rds")
    model <- readRDS("model_rf_clust.rds")
    
    # Load the original training data
    data <- read.csv("alzheimers_disease_data.csv", stringsAsFactors = TRUE)
    data$Diagnosis <- as.factor(data$Diagnosis)
    levels(data$Diagnosis) <- make.names(levels(data$Diagnosis))
    
    # Get the same preprocessing steps as in the model training
    set.seed(123)
    train_idx <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
    train_data <- data[train_idx, ]
    
    # Perform clustering on training data
    clust_train <- train_data %>% select(where(is.numeric))
    nzv <- nearZeroVar(clust_train)
    if(length(nzv) > 0) clust_train <- clust_train[, -nzv]
    clust_scaled <- scale(clust_train)
    k <- 3
    set.seed(123)
    kmeans_res <- kmeans(clust_scaled, centers = k, nstart = 25)
    train_data$Cluster <- factor(kmeans_res$cluster)
    
    debug_print("Model structure", model)
    debug_print("Training data structure", train_data)
    debug_print("Cluster levels", levels(train_data$Cluster))
    
    list(
      model = model,
      train_data = train_data,
      cluster_levels = levels(train_data$Cluster),
      kmeans_centers = kmeans_res$centers,
      clust_scaled = clust_scaled
    )
  })
  
  observeEvent(input$predict, {
    req(input$text_input)
    debug_print("Input text", input$text_input)
    
    extracted <- NULL
    err <- NULL
    tryCatch({
      extracted <- extract_all_features(input$text_input)
      debug_print("Extracted features from Python", extracted)
    }, error = function(e) {
      err <<- e$message
      debug_print("Python extraction error", err)
    })
    
    if (!is.null(err)) {
      main_message(paste0("Erreur : ", err, "\nVeuillez renseigner toutes les variables."))
      return()
    }
    
    # Get model and training data
    model_info <- model_data()
    train_data <- model_info$train_data
    
    # Convert extracted data to data frame
    new_patient <- as.data.frame(extracted, stringsAsFactors = FALSE)
    debug_print("Initial new patient data", new_patient)
    
    # Convert each column to match training data types exactly
    for (col in names(new_patient)) {
      if (col %in% names(train_data)) {
        debug_print(paste("Converting column:", col), list(
          new_value = new_patient[[col]],
          new_type = class(new_patient[[col]]),
          train_type = class(train_data[[col]]),
          train_levels = if(is.factor(train_data[[col]])) levels(train_data[[col]]) else NULL
        ))
        
        if (is.factor(train_data[[col]])) {
          new_patient[[col]] <- factor(as.character(new_patient[[col]]), 
                                     levels = levels(train_data[[col]]),
                                     ordered = is.ordered(train_data[[col]]))
        } else if (is.numeric(train_data[[col]])) {
          new_patient[[col]] <- as.numeric(new_patient[[col]])
        }
        
        debug_print(paste("After conversion:", col), list(
          new_value = new_patient[[col]],
          new_type = class(new_patient[[col]])
        ))
      }
    }
    
    # Add PatientID for clustering
    new_patient$PatientID <- 6900
    
    # Get common columns for clustering
    centers <- model_info$kmeans_centers
    debug_print("Kmeans centers", centers)
    
    common_cols <- intersect(colnames(centers), colnames(new_patient))
    debug_print("Common columns for clustering", common_cols)
    
    # Prepare numeric data for clustering
    new_patient_num <- new_patient[, common_cols, drop = FALSE]
    new_patient_num <- as.matrix(new_patient_num)
    
    # Scale the new patient data using the same scaling as training data
    new_patient_scaled <- scale(new_patient_num,
                              center = attr(model_info$clust_scaled, "scaled:center"),
                              scale = attr(model_info$clust_scaled, "scaled:scale"))
    
    debug_print("Scaled numeric data for clustering", new_patient_scaled)
    
    # Assign cluster
    clust_num <- assign_cluster(new_patient_scaled, centers)
    debug_print("Assigned cluster number", clust_num)
    
    new_patient$Cluster <- factor(clust_num, levels = model_info$cluster_levels)
    debug_print("Final new patient data with cluster", new_patient)
    
    # Make prediction
    debug_print("Attempting prediction with model", list(
      model_type = class(model_info$model),
      new_data_structure = str(new_patient),
      train_data_structure = str(train_data)
    ))
    
    pred_prob <- predict(model_info$model, new_patient, type = "prob")
    pred_class <- predict(model_info$model, new_patient)
    
    debug_print("Prediction results", list(
      probabilities = pred_prob,
      class = pred_class
    ))
    
    pred_label <- as.character(pred_class)
    # Map X0/X1 to Oui/Non
    if (pred_label == "X1") {
      result <- "Oui (Maladie d'Alzheimer)"
      # Optionally add probability here if needed for Yes case
    } else if (pred_label == "X0") {
      prob_future_alzheimer <- round(pred_prob[,2] * 100, 2) # Use prob of X1
      result <- paste0("Non (Pas d'Alzheimer)", " - Probabilité de l'avoir dans le futur : ", prob_future_alzheimer, "%") # Update text
    } else {
      result <- pred_label
    }
    main_message(result)
  })

  output$main_output <- renderText({
    main_message()
  })
}

shinyApp(ui, server)












