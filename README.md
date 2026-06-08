# Concurrent KNN

A multithreaded implementation of the **K-Nearest Neighbors (KNN)** classification algorithm written in C, using **POSIX Threads (pthreads)** for parallelism. Tested on the [MNIST handwritten digit dataset](https://drive.google.com/file/d/1Tjm3IzKqq9uMoqHcSsWKff5q7khairkL/view?usp=sharing).

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Building](#building)
- [Usage](#usage)
- [License](#license)

---

## Features

- Parallel distance computation across configurable thread count
- CSV-based dataset loading (compatible with the MNIST CSV format)
- Configurable number of neighbors (K), threads, and label count

---

## Prerequisites

- GCC (or any C99-compatible compiler)
- POSIX Threads (`pthreads`) — standard on Linux/macOS
- GNU Make

---

## Dataset Setup

The program expects two CSV files in a `data/` directory relative to the binary:

```
data/
├── mnist_train.csv   # Training split
└── mnist_test.csv    # Test split
```

Download the MNIST CSV dataset from the link below and place the files in `data/`:

> https://drive.google.com/file/d/1Tjm3IzKqq9uMoqHcSsWKff5q7khairkL/view?usp=sharing

---

## Configuration

All tunable parameters are defined as constants near the top of `Trabalho_170050432.c`:

| Constant        | Default                    | Description                              |
|-----------------|----------------------------|------------------------------------------|
| `DATA_TRAIN`    | `"data/mnist_train.csv"`   | Path to the training CSV file            |
| `DATA_TEST`     | `"data/mnist_test.csv"`    | Path to the test CSV file                |
| `TRAIN_SAMPLES` | `60000`                    | Number of training samples to load       |
| `TEST_SAMPLES`  | `10000`                    | Number of test samples to classify       |
| `N_LABELS`      | `10`                       | Number of distinct class labels          |
| `N_THREADS`     | `8`                        | Number of worker threads                 |

---

## Building

```bash
make trabalho
```

This produces a `Trabalho` binary in the project root.

---

## Usage

```bash
./Trabalho
```

Ensure the `data/` directory with the dataset files is present in the same directory as the binary before running.

---

## License

This project is provided for educational purposes. Contributions and adaptations are welcome.
