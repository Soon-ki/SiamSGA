import os


def main():
    data_path= '/media/soonki/data/LaSOT/LaSOTBenchmark'
    with open('train_id.txt','w+') as f:
        for classes in os.listdir(data_path):
            for sub_class in os.listdir(os.path.join(data_path, classes)):
                f.writelines(sub_class+'\n')


if __name__ == "__main__":
    main()
