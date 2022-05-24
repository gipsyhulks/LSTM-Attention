from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from attention_utils import get_activations, get_data_recurrent , get_data_recurrent2
from sklearn.tests.test_base import K
import numpy as np
import matplotlib.pyplot as plt

# INPUT_DIM = 1
# TIME_STEPS = 168
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

# 数据颗粒度是小时，时常是一个月，用过去168小时数据预测未来24小时数据走向
n_in = 168  # 历史数量 168个小时，7天，正好一周 这是个人选取的长度，如果感觉不妥可以修改
n_out = 24  # 预测数量 这是我们的需求，我们需要预测未来一天的流量变化，所以就是24小时
n_features = 1  # 只有一个特征
# n_test = 1
n_val = 1  # 验证的数据量
n_epochs = 20  # 迭代多少次


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    # input_dim = 1
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # a_probs = a
    output_attention_mul = merge([inputs, a_probs], output_shape=24, name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    # 加了attention的lstm模型
    inputs = Input(shape=(n_in, n_features,))
    lstm_out = LSTM(24, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(24, activation='relu')(attention_mul)
    model = Model(inputs, output)
    return model


# 导入数据，我们用不到时间戳，所以时间特征不用处理
def load_stw_data() -> pd.DataFrame:
    df_stw = pd.read_excel('north office co2 2-11月 1h.xlsx', engine='openpyxl')
    df_stw.columns = ['CollectingDate', 'Data']  # 这里的命名随意，按照兴趣
    return df_stw


def minmaxscaler(data: pd.DataFrame) -> pd.DataFrame:
    volume = data.Data.values
    volume = volume.reshape(len(volume), 1)
    # fit_transform是归一化和标准化的一个方法，X=(x-\mu（均值）)/\sigma（方差）
    # 方差计算公示是各数据与它们的均值的差的平方的均值，利用一批数据x的均值和方差来得到新的数据集X
    # X这批数据具有很好的稳定性，波动也小在最后机器学习的过程可以更快拟合，稍微提高准确率
    # 此外fit_transform函数的输入比较严格，所以需要做一下reshape,单纯的list是不行的
    volume = scaler.fit_transform(volume)
    # volume.reshape(x，y)函数是将volume转换成x行y列的数据，方便用于处理数据，比如之前手写数字
    volume = volume.reshape(len(volume), )
    data['Data'] = volume
    return data


def split_data(x, y, n_test: int):
    # python 中元组[:x]表示从第一个元素（包含）开始的x个元素，[y:]表示从下标为y的元素（包含）到最后一个的元素（包含）
    # -1 - 24 + 1 = -24 x[:-24]表示从第一个元素（包含）到倒数第24个元素（不包含）
    x_train = x[:-n_val-n_out+1]  # 这里为啥要-n_out + 1，因为这些数据会出现在我们的验证中，如果这些数据训练过了，那验证就不准确了。
    # -1:  x[-1:]表示取倒数1个数
    x_val = x[-n_val:]
    # :-1 - 24 + 1 = -24
    y_train = y[:-n_val-n_out+1]
    # -1:
    y_val = y[-n_val:]
    return x_train, y_train, x_val, y_val


# 划分X和Y，最通俗的就是几进几出，进是x的维度，出是y的维度，in是训练时常168，out是验证时常24
def build_train(train, n_in, n_out):
    # 这里删掉了时间的那列BillingDate部分
    train = train.drop(["CollectingDate"], axis=1)
    X_train, Y_train = [], []
    # train.shape[0]是数组的行数，这里是744行 其中的计算时744 - 168 - 24 + 1 = 553，迭代553 i从0-552
    for i in range(train.shape[0]-n_in-n_out+1):
        X_train.append(np.array(train.iloc[i:i+n_in]))  # iloc[0:0+168]
        # 对于这里的代码不清楚的话，可以运行看看，iloc[0+168:0+168+24]["VolumnHL"] 取VolumnHL [168,192)的数据
        Y_train.append(np.array(train.iloc[i+n_in:i+n_in+n_out]["Data"]))

    # 最终变成这两个数组返回,其中
    # X_train是一个二维数组，每个数组是VolumnHL的一个元素[[670718][648975]...[706236]]
    # Y_train是一个一维数组，存放的是VolumnHL的元素[670718,648975,...,706236]
    return np.array(X_train), np.array(Y_train)

if __name__ == '__main__':


    N = 300000
    # N = 300 -> too few = no training
    # inputs_1, outputs = get_data_recurrent2(N, TIME_STEPS, INPUT_DIM)

    # if APPLY_ATTENTION_BEFORE_LSTM:
    # m = model_attention_applied_before_lstm()
    # else:
    m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    print(m.summary())

    # m.fit([inputs_1], outputs, epochs=2, batch_size=64, validation_split=0.1)

    # 加载excel文件数据
    data = load_stw_data()
    # 数据归一化，方便拟合和提高数据精度
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 从excel中读取数据并将数据进行归一化处理，然后将2列去掉时间列，只留数据列
    data = minmaxscaler(data)

    data_copy = data.copy()
    x, y = build_train(data_copy, n_in, n_out)
    x_train, y_train, x_val, y_val = split_data(x, y, n_val)
    # model = build_lstm2(n_in, 1)
    # m = m.fit(x_train, y_train, x_val, y_val, 1)
    # 到上面为止，我们的模型就训练好了，下面就可以用模型进行预测

    m.fit(x_train, y_train, epochs=n_epochs, batch_size=128, verbose=1,  validation_data=(x_val, y_val))
    # m.fit(x_train, y_train, epochs=n_epochs, batch_size=128, validation_split=0.1)
    # m = m.evaluate(x_val, y_val)  # 计算出误差度然后输出

    predict = m.predict(x_val)

    # 将预测值反归一化，用于和实际数据的比较，scaler这个是归一化的标准，是一开始设定好的，我们归一化的时候也是采用这个标准
    validation = scaler.inverse_transform(predict)[0]  # 预测值的反归一化
    actual = scaler.inverse_transform(y_val)[0]  # 实际值的反归一化
    predict = validation
    actual = actual
    x = [x for x in range(n_out)]
    # 设置画布宽10，高5，像素300
    # plt.sca(ax1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi = 100)
    # x轴显示，线条宽度2，标签显示是predict，显示的是上面的predict数据，也就是真实值
    ax.plot(x, predict, linewidth=2.0, label = "predict")
    ax.plot(x, actual, linewidth=2.0, label = "actual")
    # 标签的显示位置，2是左上
    ax.legend(loc=2)
    # ax.set_title(bf_name)
    # 坐标的上下限
    plt.ylim((0, 2000))
    # 图中网格线设置
    plt.grid(linestyle='--')
    plt.xlabel('Time/h')
    # plt.ylabel('CO₂/ppm')
    # plt.ylabel('TVOC/ppb')
    plt.ylabel('PM2.5/μg/m³')
    # plt.show()

    # 下面这段是attention的显示，显示哪段时间步是attention的对象
    attention_vectors = []
    for i in range(1):
        testing_inputs_1 = x_val
        testing_outputs = y_val
        # testing_inputs_1, testing_outputs = get_data_recurrent2(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=False,
                                                   layer_name='attention_vec')[0], axis=1).squeeze()
        print('attention =', attention_vector)
        # assert(np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.

    import pandas as pd
    # plt.sca(ax1)
    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                              'a function of input'
                                                                               ' timesteps.')
    plt.xlabel('Time/h')
    plt.ylabel('Weight')
    plt.show()


