import os
import pandas as pd

def main():

    path = os.getcwd() + '/DataMining/twitter_data/test2017.tsv'
    df = pd.DataFrame(data=pd.read_csv(path,sep='\t',nrows=5100))

    path = os.getcwd() + '/DataMining/twitter_data/test2017_b.tsv'
    df.to_csv(path,sep='\t',index=False)


if __name__ == '__main__':
    main()
