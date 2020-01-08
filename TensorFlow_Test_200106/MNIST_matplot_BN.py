import matplotlib.pyplot as plt
from MNIST_withBN import *

labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, is_training:False })

fig = plt.figure()

for i in range(10):
    # 2행 5열의 그래프 만들고, i+1번째에 숫자 이미지를 출력
    subplot = fig.add_subplot(2, 5, i+1)
    # 이미지를 깨끗하게 출력하기 위해 x와 y의 눈금을 출력하지 않음
    subplot.set_xticks([])
    subplot.set_yticks([])
    # 출력한 이미지 위에 예측한 숫자를 출력
    # np.argmax는 tf.argmax와 같은 기능의 함수
    # 결과값인 labels의 i번째 요소가 ont-hot 인코딩 형식으로 되어 있으므로,
    # 해당 배열에서 가장 높은 값을 가진 인덱스를 예측한 숫자로 출력
    subplot.set_title('%d' % np.argmax(labels[i]))
    # 1차원 배열로 되어 있는 i번째 이미지 데이터를 28x28 형식의 2차원 배열로 변형하여 이미지 형태로 출력
    # cmap 파라미터를 통해 이미지를 그레이스케일로 출력
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()

