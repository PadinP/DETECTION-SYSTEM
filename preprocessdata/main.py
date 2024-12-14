import preprocesssing as pre

data = 'D:/CUJAE/Python/datasets-tesis/CTU-13-Dataset/escenarios/*[0123456789].binetflow'
scalers = {'minmax'}  # {'standard', 'minmax', 'robust', 'max-abs'}
samplers = ['balanced', 'smote']  # 'under_sampling', 'over_sampling', 'smote', 'svm-smote' 'adasyn'


def main():
    pre.preprocessing(data, scalers, samplers)


if __name__ == '__main__':
    main()
