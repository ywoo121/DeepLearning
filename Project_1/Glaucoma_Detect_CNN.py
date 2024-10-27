import numpy as np
import pandas as pd
import os
import json
from PIL import Image
import re    
import matplotlib.pyplot as plt

class CustomDataset:
    def __init__(self, base_dir):
        """
        CustomDataset 클래스 초기화 함수
        - 기능: 데이터 경로를 설정하고 데이터 로딩을 위한 변수들을 초기화
        - 매개변수: base_dir (str) - 데이터셋의 기본 디렉토리 경로
        """
        self.image_dir = os.path.join(base_dir, "Images/Images")
        self.json_dir = os.path.join(base_dir, "Images/json_files")
        self.label_file = os.path.join(base_dir, "G1020.csv")
        self.labels_df = None
        self.X = None
        self.y = None

    def load_data(self):
        """
        레이블 데이터를 불러오는 함수
        - 기능: CSV 파일을 읽어와서 이미지 ID와 라벨을 추출하여 저장
        - 매개변수: 없음
        - 리턴값: 없음
        """
        self.labels_df = pd.read_csv(self.label_file)
        self.labels_df['imageID'] = self.labels_df['imageID'].apply(lambda x: int(re.search(r'\d+', x).group()))
        self.X = self.labels_df['imageID'].values
        self.y = self.labels_df['binaryLabels'].values

    def custom_train_test_split(self, test_size=0.2, random_state=None, shuffle=True):
        """
        데이터셋을 훈련 및 테스트 세트로 분리하는 함수
        - 기능: 데이터를 무작위로 섞고 주어진 비율로 훈련 및 테스트 세트로 분할
        - 매개변수: 
            - test_size (float) - 테스트 세트 비율 (기본값 0.2)
            - random_state (int) - 무작위 시드 값 (기본값 없음)
            - shuffle (bool) - 셔플 여부 (기본값 True)
        - 리턴값: X_train, X_test, y_train, y_test - 분할된 데이터셋
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = len(self.X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        test_size = int(n_samples * test_size)
        train_size = n_samples - test_size

        X_train = self.X[indices[:train_size]]
        X_test = self.X[indices[train_size:]]
        y_train = self.y[indices[:train_size]]
        y_test = self.y[indices[train_size:]]

        return X_train, X_test, y_train, y_test

    def calculate_vertical_diameter(self, points):
        """
        주어진 좌표의 수직 직경을 계산하는 함수
        - 기능: `y` 좌표의 최댓값과 최솟값의 차이를 계산하여 수직 직경을 구함
        - 매개변수: points (list) - (x, y) 좌표의 리스트
        - 리턴값: 수직 직경 (float)
        """
        y_coords = [y for x, y in points]
        return max(y_coords) - min(y_coords)

    def calculate_cdr(self, json_path):
        """
        CDR (Cup-to-Disc Ratio)을 계산하는 함수
        - 기능: disc와 cup 직경을 불러와 CDR 값을 계산
        - 매개변수: json_path (str) - JSON 파일 경로
        - 리턴값: CDR 값 (float)
        """
        with open(json_path, 'r') as file:
            annotation = json.load(file)

        od_diameter, oc_diameter = 0, 0
        for shape in annotation['shapes']:
            points = shape['points']
            if shape['label'] == 'disc':
                od_diameter = self.calculate_vertical_diameter(points)
            elif shape['label'] == 'cup':
                oc_diameter = self.calculate_vertical_diameter(points)

        return oc_diameter / od_diameter if od_diameter > 0 else 0

    def preprocess_vcdr(self, vcdr_value, min_val=0, max_val=1):
        """
        CDR 값 정규화
        - 기능: 주어진 CDR 값을 0과 1 사이의 범위로 정규화
        - 매개변수:
            - vcdr_value: CDR 값
            - min_val: 최소값 (기본값: 0)
            - max_val: 최대값 (기본값: 1)
        - 리턴값: 정규화된 CDR 값
        """
        return (vcdr_value - min_val) / (max_val - min_val) if min_val <= vcdr_value <= max_val else vcdr_value

    def load_image_and_vcdr(self, image_id, img_size=(64, 64)):
        """
        이미지와 CDR 값 불러오기
        - 기능: 주어진 이미지 ID에 해당하는 이미지와 CDR 값을 불러오고 전처리
        - 매개변수:
            - image_id: 이미지 ID
            - img_size: 이미지 크기 (기본값: 64x64)
        - 리턴값: 전처리된 이미지 배열, 정규화된 CDR 값
        """
        img_path = os.path.join(self.image_dir, f"image_{image_id}.jpg")
        json_path = os.path.join(self.json_dir, f"image_{image_id}.json")

        img = Image.open(img_path).convert('L').resize(img_size)
        img = np.array(img) / 255.0

        vcdr = self.calculate_cdr(json_path)
        normalized_vcdr = self.preprocess_vcdr(vcdr)

        return img, normalized_vcdr

    def prepare_data(self, X_ids):
        """
        이미지 데이터 전처리 준비
        - 기능: 주어진 이미지 ID 리스트에 해당하는 이미지 배열과 CDR 값 반환
        - 매개변수:
            - X_ids: 이미지 ID 리스트
        - 리턴값: 전처리된 이미지 배열과 CDR 값 배열
        """
        X, vCDR = [], []
        for image_id in X_ids:
            img, vcdr = self.load_image_and_vcdr(image_id)
            X.append(img)
            vCDR.append(vcdr)
        return np.expand_dims(np.array(X), axis=-1), np.array(vCDR)
    
class SimpleCNN:
    def __init__(self, input_shape, learning_rate=0.001, reg_lambda=0.001):
        """
        SimpleCNN 클래스 초기화
        - 기능: 모델의 입력 형태, 학습률, L2 정규화 계수, 가중치 초기화
        - 매개변수:
            - input_shape: 입력 데이터의 형태
            - learning_rate: 학습률 (기본값: 0.001)
            - reg_lambda: L2 정규화 계수 (기본값: 0.001)
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.weights_conv1 = np.random.randn(3, 3, 1) * np.sqrt(1.0 / (3 * 3))  # Xavier 초기화
        self.flat_shape = 961
        self.weights_dense = np.random.randn(self.flat_shape + 1) * np.sqrt(1.0 / (self.flat_shape + 1))  # Xavier 초기화
        self.bias_dense = np.random.randn(1) * 0.01

    def conv2d(self, input_image, kernel, stride=1, padding=0):
        """
        2D 합성곱 연산
        - 기능: 입력 이미지에 대해 2D 합성곱 연산 수행
        - 매개변수:
            - input_image: 입력 이미지 배열
            - kernel: 필터 커널 배열
            - stride: 스트라이드 (기본값: 1)
            - padding: 패딩 (기본값: 0)
        - 리턴값: 합성곱 연산 결과 배열
        """
        kernel_size = kernel.shape[0]
        output_size = (input_image.shape[0] - kernel_size + 2 * padding) // stride + 1
        output = np.zeros((output_size, output_size))
        padded_image = np.pad(input_image, [(padding, padding)], mode='constant')

        for i in range(output_size):
            for j in range(output_size):
                region = padded_image[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                output[i, j] = np.sum(region * kernel)

        return output

    def relu(self, x):
        """
        ReLU 활성화 함수
        - 기능: ReLU 함수로 음수 값 0으로 변환
        - 매개변수:
            - x: 입력 배열
        - 리턴값: ReLU 적용된 배열
        """
        return np.maximum(0, x)

    def max_pool(self, input_image, pool_size=2, stride=2):
        """
        최대 풀링 연산
        - 기능: 입력 이미지에 대해 최대 풀링 연산을 수행하여 크기를 줄임
        - 매개변수:
            - input_image: 입력 이미지 배열
            - pool_size: 풀링 크기 (기본값: 2)
            - stride: 풀링 스트라이드 (기본값: 2)
        - 리턴값: 풀링이 적용된 배열
        """
        output_size = input_image.shape[0] // stride
        output = np.zeros((output_size, output_size))

        for i in range(0, input_image.shape[0], stride):
            for j in range(0, input_image.shape[1], stride):
                output[i // stride, j // stride] = np.max(input_image[i:i + pool_size, j:j + pool_size])

        return output

    def dense_layer(self, input_vector, weights, bias):
        """
        밀집층 (Dense Layer) 연산
        - 기능: 입력 벡터에 대해 가중치와 바이어스를 사용한 밀집층 연산 수행, 시그모이드 활성화 함수 사용
        - 매개변수:
            - input_vector: 평탄화된 입력 벡터
            - weights: 밀집층 가중치
            - bias: 밀집층 바이어스
        - 리턴값: 활성화 함수가 적용된 출력 값
        """
        return 1 / (1 + np.exp(-(input_vector.dot(weights) + bias)))

    def mean_squared_error(self, y_true, y_pred):
        """
        평균 제곱 오차 (MSE) 손실 함수
        - 기능: 예측값과 실제값 간의 평균 제곱 오차를 계산하고 L2 정규화를 추가하여 손실 계산
        - 매개변수:
            - y_true: 실제 라벨 값
            - y_pred: 예측 값
        - 리턴값: 평균 제곱 오차 손실 값
        """
        return np.square(y_true - y_pred).mean() + self.reg_lambda * np.sum(np.square(self.weights_dense))  # L2 정규화 추가

    def forward(self, X, vCDR):
        """
        전방향 전달 (Forward Pass)
        - 기능: 입력 이미지와 CDR 값을 통해 합성곱, ReLU, 풀링, 밀집층 순으로 전방향 전달 수행
        - 매개변수:
            - X: 입력 이미지
            - vCDR: CDR 값
        - 리턴값: 최종 출력 값 (예측 확률)
        """
        conv1 = self.conv2d(X.squeeze(), self.weights_conv1)
        relu1 = self.relu(conv1)
        pool1 = self.max_pool(relu1)
        flat = pool1.flatten()
        input_vector = np.append(flat, vCDR)
        return self.dense_layer(input_vector, self.weights_dense, self.bias_dense)

    def train(self, X_train, vCDR_train, y_train, X_val, vCDR_val, y_val, epochs, decay=0.9, batch_size=32):
        """
        모델 훈련
        - 기능: 주어진 훈련 데이터로 모델을 학습하고, 에포크마다 손실 값과 검증 정확도 계산
        - 매개변수:
            - X_train: 훈련 이미지 데이터
            - vCDR_train: 훈련 CDR 데이터
            - y_train: 훈련 라벨
            - X_val: 검증 이미지 데이터
            - vCDR_val: 검증 CDR 데이터
            - y_val: 검증 라벨
            - epochs: 훈련 에포크 수
            - decay: 학습률 감소 계수 (기본값: 0.9)
            - batch_size: 배치 크기 (기본값: 32)
        - 리턴값: 훈련 손실 목록, 검증 정확도 목록
        """
        train_losses, val_accuracies = [], []
        n_samples = len(X_train)

        for epoch in range(epochs):
            total_loss = 0
            batch_indices = np.random.permutation(n_samples)  # 샘플 순서를 무작위로 섞기
            for i in range(0, n_samples, batch_size):
                batch_idx = batch_indices[i:i+batch_size]
                X_batch, vCDR_batch, y_batch = X_train[batch_idx], vCDR_train[batch_idx], y_train[batch_idx]
                
                batch_loss = 0
                for j in range(len(X_batch)):
                    y_pred = self.forward(X_batch[j], vCDR_batch[j])
                    loss = self.mean_squared_error(y_batch[j], y_pred)
                    batch_loss += loss

                    # Backpropagation 및 가중치 업데이트
                    input_vector = np.append(self.max_pool(self.relu(self.conv2d(X_batch[j].squeeze(), self.weights_conv1))).flatten(), vCDR_batch[j])
                    self.weights_dense -= self.learning_rate * (y_pred - y_batch[j]) * input_vector
                    self.bias_dense -= self.learning_rate * (y_pred - y_batch[j])

                total_loss += batch_loss / batch_size

            # 학습률 감소
            self.learning_rate *= decay

            # 검증 정확도 계산
            correct_predictions = sum(1 for i in range(len(X_val)) if (self.forward(X_val[i], vCDR_val[i]) >= 0.5) == y_val[i])
            accuracy = correct_predictions / len(X_val)

            train_losses.append(total_loss / (n_samples / batch_size))
            val_accuracies.append(accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]}, Validation Accuracy: {accuracy}")

        return train_losses, val_accuracies
    

def plot_results(train_losses, val_accuracies, epochs):
    """
    훈련 결과 시각화
    - 기능: 훈련 손실과 검증 정확도를 플롯하여 시각적으로 표시
    - 매개변수:
        - train_losses: 훈련 손실 목록
        - val_accuracies: 검증 정확도 목록
        - epochs: 에포크 수
    - 리턴값: 없음
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()
    
    # 메인 실행 부분
if __name__ == "__main__":
    base_dir = "/kaggle/input/g1020-data/G1020"
    dataset = CustomDataset(base_dir)
    dataset.load_data()

    X_train_ids, X_val_ids, y_train, y_val = dataset.custom_train_test_split(test_size=0.2, random_state=42)

    X_train, vCDR_train = dataset.prepare_data(X_train_ids)
    X_val, vCDR_val = dataset.prepare_data(X_val_ids)

    model = SimpleCNN(input_shape=(64, 64, 1), learning_rate=0.0005)
    epochs = 10
    train_losses, val_accuracies = model.train(X_train, vCDR_train, y_train, X_val, vCDR_val, y_val, epochs)

    plot_results(train_losses, val_accuracies, epochs)