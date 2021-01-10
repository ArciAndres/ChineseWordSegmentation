from argparse import ArgumentParser
import sys
import pickle



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()

def predict(input_path, output_path, resources_path):
    verbose = True
    verboseprint = print if verbose else lambda *a, **k: None
    print("==================================================================")
    print("---------------------- Chinese word segmenter ------------------ ")
    print("==================================================================\n")
    print("> Input file: " , input_path)
    print("> Output file: " , output_path)
    print("> Resources path: " , resources_path,'\n')
    
    print("[MAIN] Modules initialization... (could take a couple of seconds)... " , end = '')
    #resources_path = '../resources/'
    
    # Add the resources folder the path so processing files can be accessed
    sys.path.insert(0, resources_path)
    import ChinesePreprocessPredict as CP
    import RebuildHelper
    import ModelConfiguration
    
    print("Done.\n","")


        
    print("[INFO] Loading vocabulary... ",end = '')
    # Load the saved vocabulary of all the unigrams and bigrams in training datasets
    vocabulary = pickle.load(open(resources_path+'/Model/vocab_all.pkl','rb'))
    print("Done.\n","")

    #Load the model with the default values, and the trained weights
    maxlength = 50
    model = ModelConfiguration.get_model_custom(len(vocabulary), maxlength=maxlength)
    print("[INFO] Loading model for prediction... (could take several seconds)... ", end = '')
    #model.load_weights(resources_path + "/Model/20190424__GSA1.1.4_all0.3_adam_lr0.001_drLS0.6_drRec0.0_emb164_emb264_batch512_units256_maxl50_weigths.h5")
    model.load_weights(resources_path + "/Model/20190424__Final2.1.1(GPU)_all100_adam_lr0.0005_drLS0.4_drRec0.4_emb164_emb264_batch512_units256_maxl50.hdf5")
    print("Done.\n")
        
    verboseprint("[INFO] Preprocessing input file... ", end = '')
    input_data = CP.ChinesePreprocess(FILE_PATH=input_path, vocabulary=vocabulary,maxlength=maxlength, verbose=False)
    sents=input_data.sents_orig
    print("Done.\n")
    
    verboseprint("[INFO] Predicting segmentation of words (it might take several seconds, or minutes)... ", end = '')
    ypred = model.predict([input_data.unigrams_pad, input_data.bigrams_pad])     
    print("Done.\n")
    
    print("[INFO] Formatting prediction into BIES form... " ,end = '')
    ybies = CP.format_prediction(ypred, input_data.sents_ready)
    result = RebuildHelper.rebuild_sentences_from_bies(ybies, input_data.sents_cut, input_data.cuts)
    print("Done.\n")

    print("[INFO] Saving file in " + output_path +" ... ", end = '')
    with open(output_path,'w', encoding='utf-8') as file:
        file.writelines('\n'.join(result))
    print("Done.\n")
    
    print("========================= Process completed ===========================")   

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
    #predict("../sample_files/msr_test_from_gold.utf8", "../outputcityu.txt",  '../resources/')

#%%
