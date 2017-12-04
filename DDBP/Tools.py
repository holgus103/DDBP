import tensorflow as tf;

def initializeOptimizer(opt, vars):
    to_init = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in vars];
    if(type(opt) is tf.train.AdamOptimizer):
        to_init.extend(s for s in list(opt._get_beta_accumulators()) if s is not None);    
    
    return tf.variables_initializer([s for s in to_init if s is not None]);
