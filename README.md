# Concurrent-KNN

Concurrent implementation of the KNN algorithm in C language using the Pthreads library.

The algorithm expects two .csv files, one corresponding to the training database and the other corresponding to the test database.

The file path is defined by the variables DATA_TRAIN and DATA_TEST (lines 21 and 25 of the file Trabalho_170050432.c);

The number of training and test samples is defined by the TRAIN_SAMPLES and TEST_SAMPLES variables (lines 22 and 26 of the Job_170050432.c file), respectively.

The number of labels in the dataset is defined by N_LABELS, in line 29.

By default, the algorithm will run in 8 threads, this value can be changed by modifying the value of the N_THREADS variable in line 34.
