import random
import pandas as pd
import numpy as np
import glob
import os
import argparse
import logging
from sklearn.model_selection import StratifiedKFold
from utils.helper import get_filename_extensions

parser = argparse.ArgumentParser(description='Create meta information')
parser.add_argument('--ffpe_only',
                    action='store_true', default=True,
                    help='keep only ffpe slides')
parser.add_argument('--cancer',
                    type=str, default='LGG',
                    help='Cancer type')
parser.add_argument('--magnification',
                    type=str, default=20,
                    help='magnification level')
parser.add_argument('--patch-size',
                    type=int, default=256,
                    help='size of the extracted patch')
parser.add_argument('--random-seed',
                    type=int, default=88,
                    help='random seed for generating the Nested-CV splits')
# K FOLDS SPLIT
parser.add_argument('--outer-fold',
                    type=int, default=5,
                    help='number of outer folds for the Nested-CV splits')
parser.add_argument('--inner-fold',
                    type=int, default=9,
                    help='number of inner folds for the Nested-CV splits')
parser.add_argument('--root',
                    type=str, default='/mnt/K/WHZ/CroMAM-main',
                    help='root directory')
parser.add_argument('--patch_root',
                    type=str, default='/mnt/H/WHZ/datasets/WSI_Process_BL20X',
                    help='patch root directory')

parser.add_argument('--stratify',
                    type=str, default='idh',
                    help='when spliting the datasets, stratify on which variable')

args = parser.parse_args()
np.random.seed(args.random_seed)
random.seed(args.random_seed)

magnification = list(args.magnification.split(','))
EXT_DATA, EXT_EXPERIMENT, EXT_SPLIT = get_filename_extensions(args)
logging_file = '%s/logs/meta_log_%s_%s.csv' % (args.root, EXT_DATA[0], EXT_EXPERIMENT)
handlers = [logging.FileHandler(logging_file, mode='w'), logging.StreamHandler()]
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=handlers)

for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %s" % (arg, value))

#负责读取或生成补丁元数据DataFrame。
def get_patch_meta(patch_dir, ext, root_dir):
    if os.path.exists('%s/dataset/patches_meta_raw_%s.pickle' % (root_dir, ext)):#检查是否存在已处理过的元数据文件
        df_cmb = pd.read_pickle('%s/dataset/patches_meta_raw_%s.pickle' % (root_dir, ext))
    else:#开始处理补丁数据
        patch_files = glob.glob('%s/*/*/*.png' % patch_dir)#这儿读取的是某个案例的所有的patch
        patch_files = sorted(patch_files, key=lambda name: name.split('/')[-2])#按路径中的第二个元素排序，即按照TCGA-41-5651/TCGA-41-5651-01Z-00-DX1.84aa5ff7-d54c-43d7-8f03-c9012d303926
        logging.info("Number of patch files (no deal): %s" % len(patch_files))
        if len(patch_files) == 0:
            return 0
        #创建一个包含补丁文件路径的新DataFrame，并为每条记录添加两个新列：file_original和submitter_id
        df_cmb = pd.DataFrame(columns=['file'])
        df_cmb['file'] = patch_files #file列存的是所有patch绝对路径，例如：/mnt/K/WHZ/datasets/histoqc_test/cromm/LGG/10_224/TCGA-41-2571/TCGA-41-2571-01Z-00-DX1.d0a404d6-6cd0-42ab-b7c7-a9c35b9efbb9/11200/1792.png
        df_cmb['file_original'] = df_cmb.file.apply(lambda x: x.split('/')[-2])#file_original列存的是TCGA-41-2571-01Z-00-DX1.d0a404d6-6cd0-42ab-b7c7-a9c35b9efbb9
        df_cmb['submitter_id'] = df_cmb.file_original.apply(lambda x: x[:12])#submitter_id存的是TCGA-41-2571

    df_cmb_meta = df_cmb.drop_duplicates('file_original').copy()  # 删去该列重复的数据
    df_cmb_meta['slide_type'] = df_cmb_meta.file_original.apply(lambda x: x.split('-')[3])#添加一个slide_type列，该列通过解析文件原始名称来确定幻灯片类型。。。。。。。。slide_type例如为01Z

    df_cmb_meta['ffpe_slide'] = 0 #添加一个新列，默认值为0
    df_cmb_meta.loc[df_cmb_meta.slide_type.str.contains('01Z|02Z|DX'), 'ffpe_slide'] = 1  #对于那些包含特定关键字（如01Z, 02Z, DX）的记录，将其值设为1

    df_cmb = df_cmb.merge(df_cmb_meta[['file_original', 'slide_type', 'ffpe_slide']], on='file_original', how='inner')#将df_cmb与df_cmb_meta中的相关信息进行合并，以便每个补丁都带有相应的元数据

    #如果命令行参数指示只处理FFPE幻灯片，则过滤掉非FFPE的记录
    if args.ffpe_only:
        df_cmb = df_cmb.loc[df_cmb.ffpe_slide == 1].reset_index(drop=True)
    logging.info("Number of final patch files finally: %s" % len(df_cmb.submitter_id))
    logging.info(df_cmb.submitter_id.unique())
    logging.info("Number of patients in the final dataset: %s" % len(df_cmb.submitter_id.unique()))

    return df_cmb

# 用于准备分类数据，根据指定的分层变量更新数据框
def classification_data_perpare(df):
    df[args.stratify] = 0  #在DataFrame中添加一列，并初始化为0
    if args.stratify == 'idh': #基于idh_status列中的值来更新分层变量列.把对应的列中值为'm'的行的分层变量列值设置为1，其他行设置为0，并且移除该列中有缺失值的行
        df.loc[df.idh_status == 'Mutant', args.stratify] = [1 if x == 'Mutant' else 0 for x in
                                                       df[df.idh_status == 'Mutant'].idh_status.to_list()]
        df = df.loc[~df.idh_status.isna()].copy().reset_index(drop=True)
    elif args.stratify == '1p19q':
        df.loc[df.pq_status == 'codel', args.stratify] = [1 if x == 'codel' else 0 for x in
                                                      df[df.pq_status == 'codel'].pq_status.to_list()]
        df = df.loc[~df.pq_status.isna()].copy().reset_index(drop=True)
    elif args.stratify == 'sur':
        df.loc[df.sur_status == 1, args.stratify] = [1 if x == 1 else 0 for x in
                                                     df[df.sur_status == 1].sur_status.to_list()]
        df = df.loc[~df.sur_status.isna()].copy().reset_index(drop=True)
    else:
        pass
    logging.info('number of participants after excluding missing time %s' % df.shape[0])
    return df

# 对数据集进行分层的K折交叉验证，并将数据分成训练集和验证集
def random_split_by_id_compare(df_cmb, df_meta, root_dir='../'):
    df_split = pd.DataFrame()#创建一个空的DataFrame来存储分层后的数据
    p_num = pd.DataFrame(df_cmb.groupby(['submitter_id']).size(), columns=['num_patches']).reset_index(inplace=False)#统计每个submitter_id的补丁数量
    vars_to_keep = ['submitter_id', 'stratify_var']
    if args.stratify:
        df_meta['stratify_var'] = df_meta[args.stratify]
    else:
        df_meta['stratify_var'] = np.random.randint(0, 2, df_meta.shape[0])

    df = df_cmb[['submitter_id']].merge(df_meta[vars_to_keep], on='submitter_id', how='inner')
    df = df.dropna()#将df_cmb中的submitter_id列与df_meta中的vars_to_keep列进行内连接，并删除包含任何NaN值的行

    df_id = df.drop_duplicates('submitter_id').reset_index(drop=True).copy()[vars_to_keep]#去重。每一个病例对应一个标签
    logging.info("Total number of patients: %s" % df_id.shape[0])

    df_id['split'] = 0
    df_id.reset_index(drop=True, inplace=True)

    kf_outer = StratifiedKFold(args.outer_fold, random_state=args.random_seed, shuffle=True)
    #对df_id进行K折分层交叉验证，每次迭代产生一个训练集索引和验证集索引
    for i, (tr_index, val_index) in enumerate(kf_outer.split(df_id, df_id['stratify_var'])):

        #分割训练集和验证集
        logging.info("-" * 40)
        df_train = df_id.loc[df_id.index.isin(tr_index)].reset_index(drop=True)
        df_val = df_id.loc[df_id.index.isin(val_index)].reset_index(drop=True)
        logging.info("Working on outer split %s .... Train: %s; Val: %s" % (i, df_train.shape[0], df_val.shape[0]))

        #将训练集和验证集分别与p_num进行左连接，以包含每个submitter_id的补丁数量信息。
        dt = df_train.merge(p_num, on='submitter_id', how="left")
        dv = df_val.merge(p_num, on='submitter_id', how="left")
        #将训练集和验证集的相关信息添加到df_split中
        df_split[[f"fold{i}_t", f"t_nums{i}", f"t_lable{i}"]] = dt[["submitter_id", "num_patches", 'stratify_var']]
        df_split[[f"fold{i}_v", f"v_nums{i}", f"v_lable{i}"]] = dv[["submitter_id", "num_patches", 'stratify_var']]

        #将训练集和验证集的元数据保存为pickle文件
        df_train[['submitter_id', 'split']]. \
            merge(df_meta, on='submitter_id', how='inner'). \
            to_pickle(f'{root_dir}/dataset/{args.cancer}_{args.stratify}_meta_train_x{magnification[0]}_{i}.pickle')
        df_val[['submitter_id', 'split']]. \
            merge(df_meta, on='submitter_id', how='inner'). \
            to_pickle(f'{root_dir}/dataset/{args.cancer}_{args.stratify}_meta_val_x{magnification[0]}_{i}.pickle')
    #将整个分割结果保存为CSV文件
    df_split.to_csv(
        f"{args.root}/logs/{args.cancer}_{args.stratify}_data_split_{args.magnification}.csv")


if __name__ == '__main__':
    # process meta information / meta file path
    fname_meta = args.root + '/dataset/meta_files/meta_clinical_%s_%s.csv' % (args.cancer, args.stratify)
    if os.path.isfile(fname_meta):
        # read csv
        df_meta = pd.read_csv(fname_meta)
        df_meta = classification_data_perpare(df_meta)
        print(df_meta.head())
    else:
        pass
    logging.info(df_meta.describe())

    df_cmbs = []
    df_cmb = 0
    for i in range(len(EXT_DATA)):
        try:
            # patch_dir = args.patch_root + '/%s/%s_%s' % (args.cancer, magnification[i], args.patch_size)  # patches path
            patch_dir = args.patch_root  # patches path
            # process patch information
            patch_meta_file = '%s/dataset/patches_meta_%s.pickle' % (args.root, EXT_DATA[i])  # patch_meta_file path
            if os.path.exists(patch_meta_file):
                logging.info("patch meta file %s already exists!" % patch_meta_file)
                logging.info("patch num : %s " % len(pd.read_pickle(patch_meta_file)))
                df_cmbs.append(pd.read_pickle(patch_meta_file))
            else:
                df_cmbs.append(get_patch_meta(patch_dir, EXT_DATA[i], args.root))
                df_cmbs[i].to_pickle(patch_meta_file)
            p_nums = pd.DataFrame(df_cmbs[i].groupby(['submitter_id']).size(), columns=['num_patches'])#每个案例patch的数量
            p_nums.to_csv(
                f"{args.root}/logs/{args.cancer}_{args.stratify}_patch_num_{magnification[i]}.csv")
        except Exception as e:
            print(e)
    if len(EXT_DATA) != 1:
        df1_nums = df_cmbs[0].drop_duplicates('submitter_id').reset_index(drop=True)
        df2_nums = df_cmbs[1].drop_duplicates('submitter_id').reset_index(drop=True)
        if df1_nums.shape[0] > df2_nums.shape[0]:
            df_cmb = df_cmbs[1]
        elif df1_nums.shape[0] <= df2_nums.shape[0]:
            df_cmb = df_cmbs[0]
    else:
        df_cmb = df_cmbs[0]
    random_split_by_id_compare(df_cmb, df_meta, args.root)
