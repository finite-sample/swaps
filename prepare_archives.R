args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
raw <- args[[1]]
out <- args[[2]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)

required <- c("Matrix", "SparseM", "e1071")
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Install the required R package:", pkg))
  }
}

housing_env <- new.env()
load(file.path(raw, "HousingPriceFormatted.RData"), envir = housing_env)
housing <- housing_env$HousingPriceFormatted
proxy <- housing$ProxyDat
reference <- housing$GoodDat
expected <- c("HousingPrice", "Income", "Nightlights", "RoadLength")
stopifnot(identical(names(proxy), expected), identical(names(reference), expected),
          nrow(proxy) == 46418L, identical(dim(proxy), dim(reference)),
          identical(rownames(proxy), rownames(reference)),
          identical(proxy$HousingPrice, reference$HousingPrice),
          identical(proxy$Income, reference$Income),
          all(is.finite(as.matrix(proxy))), all(is.finite(as.matrix(reference))))
write.csv(data.frame(y = proxy$HousingPrice, income = proxy$Income,
                    proxy_nightlights = proxy$Nightlights,
                    proxy_roadlength = proxy$RoadLength,
                    reference_nightlights = reference$Nightlights,
                    reference_roadlength = reference$RoadLength),
          file.path(out, "housing.csv"), row.names = FALSE)

train <- read.csv(file.path(raw, "unciviltraining.csv"), stringsAsFactors = FALSE)
validation <- read.csv(file.path(raw, "uncivilvalidation.csv"),
                       stringsAsFactors = FALSE)
stopifnot(nrow(train) == 2413L, nrow(validation) == 613L,
          !anyDuplicated(train$id), !anyDuplicated(validation$id),
          !anyNA(train$id), !anyNA(validation$id),
          length(intersect(train$id, validation$id)) == 0L,
          all(train$uncivil %in% c(0, 1)),
          all(validation$uncivil %in% c(0, 1)),
          all(is.finite(validation$score)))

read_dtm <- function(name, rows) {
  coordinates <- read.csv(file.path(raw, name), header = FALSE)
  stopifnot(ncol(coordinates) == 2L, all(is.finite(as.matrix(coordinates))),
            all(as.matrix(coordinates) == floor(as.matrix(coordinates))),
            all(coordinates[[1]] >= 0 & coordinates[[1]] < rows),
            all(coordinates[[2]] >= 0 & coordinates[[2]] < 2172L))
  matrix <- Matrix::sparseMatrix(i = coordinates[[1]] + 1L,
                                j = coordinates[[2]] + 1L,
                                dims = c(rows, 2172L))
  SparseM::as.matrix.csr(matrix, nrow = rows, ncol = 2172L)
}

train_matrix <- read_dtm("unciviltrainingdt.dat", nrow(train))
validation_matrix <- read_dtm("uncivilvalidationdt.dat", nrow(validation))
weights <- c("1" = 1 / mean(train$uncivil),
             "0" = 1 / mean(1 - train$uncivil))
set.seed(8102017)
model <- e1071::svm(x = train_matrix, y = factor(train$uncivil),
                    class.weights = weights, type = "C-classification",
                    kernel = "radial", gamma = 1 / 2172, cost = 1,
                    scale = TRUE, tolerance = 0.001, shrinking = TRUE,
                    cross = 0, probability = FALSE)
predicted <- as.numeric(as.character(predict(model, validation_matrix)))
stopifnot(length(predicted) == nrow(validation), all(predicted %in% c(0, 1)))
write.csv(data.frame(y = pmax(pmin(validation$score, 10), -10),
                    proxy_incivility = predicted,
                    reference_incivility = validation$uncivil),
          file.path(out, "reddit.csv"), row.names = FALSE)
write.csv(data.frame(training_rows = nrow(train), validation_rows = nrow(validation),
                    overlapping_ids = length(intersect(train$id, validation$id)),
                    misclassified = sum(predicted != validation$uncivil),
                    true_positive = sum(predicted == 1 & validation$uncivil == 1),
                    false_positive = sum(predicted == 1 & validation$uncivil == 0),
                    false_negative = sum(predicted == 0 & validation$uncivil == 1),
                    true_negative = sum(predicted == 0 & validation$uncivil == 0)),
          file.path(out, "preparation_audit.csv"), row.names = FALSE)
writeLines(trimws(capture.output(sessionInfo()), which = "right"),
           file.path(out, "R-session.txt"))
