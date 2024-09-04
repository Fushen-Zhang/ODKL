import tensorflow as tf
def PersitenceSave(session):
    saver = tf.train.Saver()
    basepath = './checkpoints/model1'
    saver.save(session, basepath)  # 'model1.data' & meta files
    step = 1000
    saver.save(session, basepath, global_step=step)   # 'model1-1000.data' & meta files
    step = 2000
    saver.save(session, basepath, global_step=step, write_meta_graph=False)  # 'model1-2000.data' ; NO meta files

    saverWithPolicy = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    saver.save(session, basepath)  # will save every 2 hours, and keep 4 last ones

def PersistenceLoad(session):
    basepath = './checkpoints/model1'
    saver = tf.train.import_meta_graph(basepath + '.meta')  # loads the graph into the current session
    print('Warning: May restore different weights than expected:')
    saver.restore(session, tf.train.latest_checkpoint('./checkpoints/')) #loads the parameters

    g = tf.get_default_graph()
    x = g.get_tensor_by_name('x:0')
    w1 = g.get_tensor_by_name('w1:0')
    loss = g.get_tensor_by_name('loss:0')  # loss = tf.mean( ... , name='loss')
    [loss_] = session.run_stateless([loss], {x: 1})


def PersitenceSaveThis(session, name):
    saver = tf.train.Saver()
    basepath = './checkpoints/' + name
    saver.save(session, basepath)

def PersistenceLoadThis(session, name, withGraph = True, basedir = '.'):
    basepath = basedir + '/checkpoints/' + name
    if withGraph:
        saver = tf.train.import_meta_graph(basepath + '.meta')  # loads the graph into the current session
        saver.restore(session, basepath) #loads the parameters
    else:
        saver = tf.train.Saver()
        saver.restore(session, basepath)

import pickle
def SaveObject(object, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def LoadObject(name):
    try:
        with open(name + '.pkl', 'rb') as f:
            object = pickle.load(f)
        return object
    except:
        return None

