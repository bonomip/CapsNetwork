from numpy import sqrt 
import numpy as np
from tqdm import tqdm
import tensorflow as tf

def get_confusion_table_for(mat, digit, tot):
    
    dict_ = {}
    
    # condition positive (how many images were < digit >)
    p = 0
    for i in range(0, 10):
        p += mat[i, digit]
    dict_["Condition Positive"] = p
    
    #condition negative (how many images were not < digit > )
    n = tot - p
    dict_["Condition Negative"] = n 
    
    tp = mat[digit, digit]
    dict_["True Positive (TN)"] = tp
    
    fn = -tp
    for i in range(0, 10):
        fn += mat[i, digit]
    dict_["False Negative (FN)"] = fn
    
    fp = -tp
    for i in range(0, 10):
        fp += mat[digit, i]
    dict_["False Positive (FP)"] = fp 
    
    tn = tot + tp    
    for i in range(0, 10):
        tn -= mat[digit, i] + mat[i, digit]
    dict_["True Negative (TN)"] = tn 

    tpr = tp / p
    dict_["Sensitivity (TPR)"] = tpr 

    tnr = tn / n
    dict_["Specificity (TNR)"] = tnr 

    ppv = tp / (tp + fp)
    dict_["Positive Predictive Value (PPV)"] = ppv

    npv = tn / ( tn + fn )
    dict_["Negative Predictive Value (NPV)"] = npv 

    fnr = 1 - tpr
    dict_["False Negative Rate (FNR)"] = fnr 

    fpr = 1 - tnr
    dict_["False Positive Rate (FPR)"] = fpr 

    fdr = 1 - ppv
    dict_["False Discovery Rate (FDR)"] = fdr 

    for_ = 1 - npv
    dict_["False Omission Rate (FOR)"] = for_ 

    lr_p = tpr/fpr
    dict_["Positive Likelihood Ratio (LR+)"] = int(lr_p) 
    
    lr_n = fnr/tnr
    dict_["Negative Likelihood Ratio (LR-)"] = lr_n 

    pt = sqrt( fpr ) / ( sqrt(tpr) * sqrt(fpr) )
    dict_["Prevalence Threshold (PT)"] = pt 

    ts = tp / ( tp + fn + fp )
    dict_["Threat Score"] = ts 

    prevalence = p / tot
    dict_["Prevalence"] = prevalence 
    
    acc = ( tp + tn ) / tot
    dict_["Accuracy (ACC)"] = acc 

    ba = ( tpr * tnr ) / 2
    dict_["Balanced Accuracy"] = ba 

    f1 = ( 2*tp ) / ( 2*tp + fp + fn )
    dict_["F1 Score"] = f1 

    mcc = ( ( tp * tn ) - ( fp * fn ) ) / sqrt( ( tp+fp )*( tp*fn )*( tn+fp )*(tn+fn) )
    dict_["Matthews Correlation Coefficient (MCC)"] = mcc
    
    fm = sqrt( ppv * tpr )
    dict_["Fowlkes–Mallows Index (FM)"] = fm
    
    bm = tpr + tnr - 1
    dict_["Bookmaker Informedness (BM)"] = bm
    
    mk = ppv + npv - 1
    dict_["Markedness (MK)"] = mk
    
    #from 0 to inf
    dor = lr_p / lr_n
    dict_["Diagnostic odds ratio (DOR)"] = int(dor)
    
    header = np.array(list(dict_.keys()))
    values = np.array(list(dict_.values()))
    
    return values, header

def get_confusion_table(confusion_matrix, dataset_size):
    values = []
    index = []
    columns = []
    for i in range(0, 10):
        v, h = get_confusion_table_for(confusion_matrix, i, dataset_size)
        values.append(v)
        index = h
        columns.append(i)
    
    values = np.around(np.array(values), decimals=3)
    index = np.array(index)
    columns = np.array(columns)

    return values, columns, index

def get_accuracy(matrix):
    a = 0        
    for i in range(0, 10):        
        a += matrix[i, i]
    
    return a

def get_confusion_mat(predict_function, data, name):
    matrix = np.zeros((10,10))
    
    with tqdm(total=len(data)) as pbar:
        
        description = "Creating "+name+" confusion matrix"
        pbar.set_description_str(description)
        for X_batch, y_batch in data:
            
            res = predict_function(X_batch)
            for i in range(0, y_batch.shape[0]):
                
                matrix[y_batch[i], res[i]] += 1 # row, col
        
            pbar.update(1)

    return matrix

def normalize_matrix(matrix, tot):
    mat = matrix.copy()
    for i in range(0, 10):
        for j in range(0, 10):
            mat[i, j] = (mat[i, j] / tot) * 100  
    return  np.around(mat, decimals=3)

def normalize_matrix_on_row(matrix):
    mat = matrix.copy()
    for c in range(0,10):
        
        mat[:,c] /= np.sum(mat[:,c])
        mat[:,c] *= 100 
                           
    return np.around(mat, decimals=3)

def normalize_matrix_on_columns(matrix):
    mat = matrix.copy()
    for r in range(0,10):
        
        mat[r,:] /= np.sum(mat[r,:])
        mat[r,:] *= 100
                           
    return np.around(mat, decimals=3)

def print_eval_params(mat):
    
    tot_in = 0
    tot_out = 0
    
    for j in range(0, 10):
        digit = j
        
        v = 0
        for i in range(0, 10):
            v += mat[digit, i]
        k = mat[digit, digit]
        tot_out += (k/v)
        r = (k/v)*100
        
        
        #the model has predict digit v times, k of that was right
        #osservo l'output, qual'è la probabilità che il modello abbia classificato in modo corretto?
        print("Proportion of digit < "+str(digit)+" > that was correctly identified =\t\t"+str(r)+"%") #sommo sulla riga

        v = 0
        for i in range(0, 10):
            v += mat[i, digit]  
        k = mat[digit, digit]
        tot_in += (k/v)
        r = (k/v)*100
        
        #the environment has shown digit v times, the model has predicted it k times
        #osservo l'input, qual'è la probabilità che venga classificato correttamente? 
        print("Proportion of actual digit < "+str(digit)+" > which is correctly identified =\t"+str(r)+"%") #sommo sulla colonna
        
        print("")

    print("Average observation correctness respect to ouput = "+str(tot_out*10))
    print("Average observation correctness respect to input = "+str(tot_in*10))

def get_error_index(model, images, labels, off):
    error = list()
    prediction = list()

    for i in range (images.shape[0]):
        x = tf.reshape(images[i], [1, 40, 40, 1])
        y = labels[i]
        p = model.predict(x)[0]
        if p!=y:
            prediction.append(p)
            error.append(i+off)
            
    return error, prediction