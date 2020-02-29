
# install.packages("mvtnorm", repos=c("http://rstudio.org/_packages", "http://cran.rstudio.com",dependencies=TRUE))
require(C50)
require(e1071)
library(caret)

balanceCSV <- read.csv("C:\\Users\\user\\Documents\\R\\yt\\depression_detector_c5\\depression_detector\\final_c5_data.csv")
str(balanceCSV)
summary(balanceCSV)


set.seed(42);

# shuffle 
balanceDF <- balanceCSV[order(runif(nrow(balanceCSV))),]
tail(balanceDF,25)

# Classification data and its labels
X <- balanceDF[,1:6]
y <- balanceDF[,7]

# Divide into training and test data
trainX <- X[1:2426,]
trainy <- y[1:2426]
testX <- X[2427:3466,]
testy <- y[2427:3466]


# set.seed(42);

# Build model
model <- C50::C5.0( trainX, as.factor(trainy), trials=20)
summary( model )

# Predicting values
C50_prediction <- predict( model, testX, type="class" );
C50_prediction


# Confusion Matrix
table(testy,C50_predict)


# Check accuracy
accuracy <- sum( C50_predict == testy ) / length( C50_predict )
paste0((accuracy * 100), "% accuracy");

# Other metrics
cm <- confusionMatrix(factor(C50_prediction), reference = factor(testy));

cm[["byClass"]][ , "F1"] 




