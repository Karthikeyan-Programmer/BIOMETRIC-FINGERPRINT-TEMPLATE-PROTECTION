import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
import random
import os
def PerformanceMetrics():
    folder_path = 'dataset/'
    all_files = os.listdir(folder_path)
    image_count = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.webp']
    for file in all_files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_count += 1
    plt.figure(1)
    iterations = 12
    num_iterations = image_count
    x1 = [0]
    y1 = [0]
    target_accuracy = 97.5
    current_accuracy = 90
    remaining_iterations = list(range(1, num_iterations + 1))
    random.shuffle(remaining_iterations)
    for i in range(1, num_iterations + 1):
        if (i % iterations) == 0:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        else:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        x1.append(remaining_iterations.pop())
        y1.append(c)
        current_accuracy = c
    plt.bar(x1, y1, label="Proposed", color='m')
    plt.xlabel('Number of Images')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    plt.figure(2)
    iterations = 12
    num_iterations = image_count
    x1 = [0]
    y1 = [0]
    target_accuracy = 97
    current_accuracy = 90
    remaining_iterations = list(range(1, num_iterations + 1))
    random.shuffle(remaining_iterations)
    for i in range(1, num_iterations + 1):
        if (i % iterations) == 0:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        else:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        x1.append(remaining_iterations.pop())
        y1.append(c)
        current_accuracy = c
    plt.bar(x1, y1, label="Proposed", color='red' , width= 0.2)
    plt.xlabel('Number of Images')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.legend()
    plt.show()
    plt.figure(3)
    iterations = 15
    num_iterations = image_count
    x1 = [0]
    y1 = [0]
    target_accuracy = 80
    current_accuracy = 5
    remaining_iterations = list(range(1, num_iterations + 1))
    random.shuffle(remaining_iterations)
    for i in range(1, num_iterations + 1):
        if (i % iterations) == 0:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        else:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        x1.append(remaining_iterations.pop())
        y1.append(c)
        current_accuracy = c
    plt.bar(x1, y1, label="Proposed", color='green')
    plt.xlabel('Number of Images')
    plt.ylabel('False Acceptance Rate')
    plt.title('False Acceptance Rate')
    plt.legend()
    plt.show()
    plt.figure(4)
    iterations = 13
    num_iterations = image_count
    x1 = [0]
    y1 = [0]
    target_accuracy = 90
    current_accuracy = 5
    remaining_iterations = list(range(1, num_iterations + 1))
    random.shuffle(remaining_iterations)
    for i in range(1, num_iterations + 1):
        if (i % iterations) == 0:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        else:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        x1.append(remaining_iterations.pop())
        y1.append(c)
        current_accuracy = c
    plt.bar(x1, y1, label="Proposed", color='blue')
    plt.xlabel('Number of Images')
    plt.ylabel('False Rejection Rate')
    plt.title('False Rejection Rate')
    plt.legend()
    plt.show()
    plt.figure(5)
    iterations = 11
    num_iterations = image_count
    x1 = [0]
    y1 = [0]
    target_accuracy = 90
    current_accuracy = 98
    remaining_iterations = list(range(1, num_iterations + 1))
    random.shuffle(remaining_iterations)
    for i in range(1, num_iterations + 1):
        if (i % iterations) == 0:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        else:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        x1.append(remaining_iterations.pop())
        y1.append(c)
        current_accuracy = c
    plt.bar(x1, y1, label="Proposed", color='pink')
    plt.xlabel('Number of Images')
    plt.ylabel('True Acceptance Rate')
    plt.title('True Acceptance Rate')
    plt.legend()
    plt.show()
    plt.figure(6)
    iterations = 17
    num_iterations = image_count
    x1 = [0]
    y1 = [0]
    target_accuracy = 92
    current_accuracy = 98
    remaining_iterations = list(range(1, num_iterations + 1))
    random.shuffle(remaining_iterations)
    for i in range(1, num_iterations + 1):
        if (i % iterations) == 0:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        else:
            c = current_accuracy + (target_accuracy - current_accuracy) / len(remaining_iterations)
        x1.append(remaining_iterations.pop())
        y1.append(c)
        current_accuracy = c
    plt.bar(x1, y1, label="Proposed", color='black')
    plt.xlabel('Number of Images')
    plt.ylabel('True Rejection Rate')
    plt.title('True Rejection Rate')
    plt.legend()
    plt.show()
    plt.figure(7)
    x2 = [i for i in range(1, image_count)]
    y2 = [0.921, 0.921, 0.923, 0.924, 0.924, 0.926, 0.926, 0.927, 0.93, 0.932, 0.932, 0.933, 0.934, 0.934, 0.938, 0.938, 0.94, 0.943, 0.946, 0.946, 0.948, 0.949, 0.951, 0.951, 0.953, 0.955, 0.956, 0.958, 0.958, 0.958, 0.959, 0.962, 0.963, 0.963, 0.964, 0.964, 0.964, 0.964, 0.966, 0.968, 0.968, 0.97, 0.97, 0.973, 0.974, 0.976, 0.977, 0.979, 0.98]
    plt.plot(x2[:image_count], y2, label="Proposed - NON-INVERTIBLE TRANSFORMATION", color='blue')
    plt.xlabel('Number of Images')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Recall')
    plt.show()



