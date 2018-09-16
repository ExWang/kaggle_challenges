import numpy as np
import pandas as pd
import gc
import os
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

import myModel

# TRAIN_SET_PATH = "/media/sbaer/Warehouse/Dataset/kaggle/HCDR/train_df.csv"
# TRAIN_SET_PATH = "/media/sbaer/Warehouse/Dataset/kaggle/HCDR/New_Train_df.csv"
# TEST_SET_PATH = "/media/sbaer/Warehouse/Dataset/kaggle/HCDR/test_df.csv"

TRAIN_SET_PATH = "/media/sbaer/Warehouse/Dataset/kaggle/HCDR/norm_train_df.csv"
TEST_SET_PATH = "/media/sbaer/Warehouse/Dataset/kaggle/HCDR/norm_test_df.csv"

DEBUG_SET_PATH = "/media/sbaer/Warehouse/Dataset/kaggle/HCDR/train_df.csv"

CHECKPOINT_dir = "./checkpoints"

_BATCH_SIZE = 64  # 32 for SGD
_NUM_PREPRO_THREAD = 4

_ITER_EPOCH = 7

_SAVE_CKPT_ITER = 12000
_MAX_TO_KEEP = 5

_MODE = "debug"  # test  train   debug


def _get_batch(data, label):
    num_preprocess_threads = _NUM_PREPRO_THREAD
    batch_size = _BATCH_SIZE

    queue_cap = (2 * num_preprocess_threads * batch_size)
    data_queue = tf.train.slice_input_producer([data, label], shuffle=True, seed=12345)

    data_batch, label_batch = tf.train.batch(
        data_queue, batch_size=batch_size, num_threads=num_preprocess_threads, capacity=queue_cap
    )

    return data_batch, label_batch


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    print("<use focal loss>")
    print("logits:", logits)
    print("labels:", labels)
    labels = tf.cast(labels, dtype=tf.float32)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    # num_samples = tf.cast(tf.reduce_prod(tf.shape(labels)), tf.float32)

    probs = tf.sigmoid(logits)

    alpha_t = tf.ones_like(logits) * alpha
    alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
    probs_t = tf.where(labels > 0, probs, 1.0 - probs)

    weight_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)

    # Normalize
    n_pos = tf.reduce_sum(labels)

    # def has_pos():
    #     return loss / tf.cast(n_pos, tf.float32)
    #
    # def no_pos():
    #     # total_weights = tf.stop_gradient(tf.reduce_sum(weight_matrix))
    #     # return loss / total_weights
    #     return loss

    loss = tf.reduce_mean(weight_matrix * ce_loss) / n_pos

    # loss = tf.cond(n_pos > 0, has_pos, no_pos)

    return loss


def forWSH(stratified=False, num_folds=5):
    # 在这里把数据集读取进来
    # train_df 训练集
    # test_df 测试集

    train_df = None
    test_df = None

    if _MODE == "train":
        train_df = pd.read_csv(TRAIN_SET_PATH)
        train_col = train_df.columns.values
        print("Starting... Train shape: {}".format(train_df.shape))
        print(train_df.columns)

    elif _MODE == "test":
        test_df = pd.read_csv(TEST_SET_PATH)
        test_col = test_df.columns.values
        print("Starting... Test shape: {}".format(test_df.shape))
        print(test_df.columns)
    elif _MODE == "debug":
        debug_df = pd.read_csv(DEBUG_SET_PATH)
        print("Starting... Debug shape: {}".format(debug_df.shape))
        print(debug_df.columns)
    else:
        raise ValueError

    # 交叉验证模型
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                random_state=47)  # n_splits:默认3；shuffle:默认False;shuffle会对数据产生随机搅动(洗牌)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    # oof_preds = np.zeros(train_df.shape[0])
    # sub_preds = np.zeros(test_df.shape[0])

    if not os.path.exists(CHECKPOINT_dir):
        os.mkdir(CHECKPOINT_dir)
        print("Create checkpoint saving dir.")

    if _MODE == "train":

        feats = [f for f in train_df.columns if
                 f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'Unnamed: 0']]

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            # print(train_y)
            train_x = pd.concat([train_x, valid_x])
            train_y = pd.concat([train_y, valid_y])

            train_x = train_x.fillna(0)
            train_y = train_y.fillna(0)
            valid_x = valid_x.fillna(0)
            valid_y = valid_y.fillna(0)
            # print(train_y)
            train_x = train_x.values
            train_y = train_y.values
            valid_x = valid_x.values
            valid_y = valid_y.values
            # print(train_x)
            # print(train_y)

            print("----DATA SHAPE----")
            print(train_x.shape)
            print(train_y.shape)
            print(valid_x.shape)
            print(valid_y.shape)
            print("------------------")

            num_train_examples = train_x.shape[0]
            # num_val_examples = valid_x.shape[0]

            num_train_iter = int(num_train_examples / _BATCH_SIZE) + 1
            # num_val_iter = int(num_val_examples / _BATCH_SIZE) + 1

            # print(train_y)

            train_x = train_x.astype(np.float32)
            train_y = train_y.astype(np.float32)
            # valid_x = valid_x.astype(np.float32)
            # valid_y = valid_y.astype(np.int)

            print(train_y)
            train_y_ex = np.zeros((train_y.shape[0], 2), np.int)
            for index in range(train_y_ex.shape[0]):
                if train_y[index] == 1:
                    train_y_ex[index][1] = 1
                else:
                    train_y_ex[index][0] = 1
            print("===============================")
            train_y = train_y_ex
            print(train_y_ex)
            print(train_y_ex.shape)

            # data_batch, label_batch = _get_batch(train_x, train_y)
            # print(data_batch, label_batch)
            # logitis = myModel.risk_model(data_batch)
            # print(logitis)

            data_batch, label_batch = _get_batch(train_x, train_y)
            print(data_batch, label_batch)
            logitis = myModel.risk_model(data_batch)
            print(logitis)

            learning_rate = 0.0001

            # l_shape = label_batch.get_shape()
            # label_batch = tf.reshape(label_batch, [l_shape[0], 1])
            # label_batch = tf.to_float(label_batch)
            # sum_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.clip_by_value(logitis, 1e-8, 1.0),
            #                                                    labels=label_batch)
            # sum_loss = -tf.reduce_sum(label_batch * tf.log(tf.clip_by_value(logitis, 1e-8, 1.0)),
            #                           reduction_indices=[1])
            # sum_loss = -tf.reduce_sum(label_batch * tf.log(tf.clip_by_value(logitis, 1e-8, 1.0))
            #                          + (1 - label_batch) * tf.log(1 - tf.clip_by_value(logitis, 1e-8, 1.0)))
            # sum_loss = focal_loss(labels=label_batch, logits=logitis)
            sum_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_batch,
                                                                  logits=logitis)
            batch_loss = tf.reduce_mean(sum_loss)
            # batch_loss = sum_loss
            print(sum_loss)
            print(batch_loss)
            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(batch_loss)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(batch_loss)
            # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(batch_loss)
            # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(batch_loss)

            init_op = tf.global_variables_initializer()

            saver = tf.train.Saver(max_to_keep=_MAX_TO_KEEP)

            moudke_file = tf.train.latest_checkpoint(CHECKPOINT_dir)

            with tf.Session() as sess:
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                if os.path.exists(CHECKPOINT_dir):
                    try:
                        saver.restore(sess, moudke_file)
                        print(moudke_file)
                        print("Restore checkpoint from <{:s}> success!".format(moudke_file.split("/")[-1]))
                    except ValueError:
                        print("Restore checkpoint failed--->Nothing to restore")
                    finally:
                        print("------------")

                try:

                    for epoch_now in range(_ITER_EPOCH):
                        print("EPOCH:", epoch_now)
                        for iter_now in range(num_train_iter):
                            # t1 = sess.run(label_batch)
                            # print(t1)
                            ret = sess.run([data_batch])
                            _, loss_total, t1, t2 = sess.run([train_step, batch_loss, logitis, label_batch])
                            if iter_now % 100 == 0:
                                print("epoch:", epoch_now,
                                      " iter:", iter_now,
                                      " XE_loss:", loss_total)
                                # print(t1)
                                # print(t2)
                            if iter_now % _SAVE_CKPT_ITER == 0:
                                print("=====Save checkpoint====")
                                saver.save(sess,
                                           os.path.join(CHECKPOINT_dir, 'model.ckpt'),
                                           global_step=(epoch_now + 1) * 10000 + iter_now)

                except tf.errors.OutOfRangeError:
                    print("Train done.")
                finally:
                    coord.request_stop()
                coord.join(threads)
            break
        print("> Finished this train period! <")

        # 实现nn的地方！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # ！！！！！！！！！！！！！！！！！！！！！！！
        # ！！！！！！！！！！！！！！！！！！！！！！！
        # 下面举个例子
        # clf = LGBMClassifier()
        # clf.fit(train_x,
        #         train_y)
        #  另一个方法 clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc')
        # # 分割线~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # oof_preds[valid_x] = clf.predict_proba(valid_x)[:, 1]  # 要输出概率
        # sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits  # 要输出概率

    if _MODE == "test":
        print("Test mode!")
        # print(test_df)
        feats = [f for f in test_df.columns if
                 f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        print(test_df.shape)
        # test_df = train_df
        test_x = test_df[feats]
        test_x = test_x.fillna(0)
        test_x = test_x.values
        test_x = test_x.astype(np.float32)
        print(type(test_x), test_x.shape)

        g = tf.Graph()
        with g.as_default():
            input = tf.placeholder(dtype=tf.float32,
                                   shape=[1, test_x.shape[1]],
                                   name="data_feed")

            print(input)
            output = myModel.risk_model(input, mode=_MODE)
            predict_op = tf.nn.sigmoid(output)
            saver = tf.train.Saver()
        g.finalize()

        moudke_file = tf.train.latest_checkpoint(CHECKPOINT_dir)

        with tf.Session(graph=g) as sess:
            if os.path.exists(CHECKPOINT_dir):
                try:
                    saver.restore(sess, moudke_file)
                    print(moudke_file)
                    print("Restore checkpoint from <{:s}> success!".format(moudke_file.split("/")[-1]))
                except ValueError:
                    print("Restore checkpoint failed--->Nothing to restore")
                finally:
                    print("------------")
            res_list = []
            for one in range(test_x.shape[0]):
                # print(one)
                one_data_fake = test_x[one]
                # print(one_data_fake.shape)
                one_data = np.reshape(one_data_fake, [1, one_data_fake.shape[0]])
                # print(one_data.shape)
                predicted_logitis, predict_percentage = sess.run(
                    [output, predict_op],
                    feed_dict={"data_feed:0": one_data})
                print(one, predicted_logitis, predict_percentage)
                ready_to_rec = predict_percentage[0][1]
                # if ready_to_rec>0.5:
                #     ready_to_rec = 1
                # else:
                #     ready_to_rec = 0
                # res_list.append("{:.1f}".format(float(ready_to_rec)))
                res_list.append(ready_to_rec)
                # Need to save
        lab = test_df['SK_ID_CURR']
        dataframe = pd.DataFrame({'SK_ID_CURR': lab, 'TARGET': res_list})
        dataframe.to_csv("results_test.csv", index=False, sep=',')
    # print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    # del clf, train_x, train_y, valid_x, valid_y
    gc.collect()

    # test_df['TARGET'] = sub_preds
    # test_df[['SK_ID_CURR', 'TARGET']].to_csv("记得起个名字", index=False)
    #
    # train_df['mode记得起个名字'] = oof_preds
    # train_df[['SK_ID_CURR', 'mode7']].to_csv("mode记得起个名字.csv", index=False)


if __name__ == "__main__":
    forWSH()
