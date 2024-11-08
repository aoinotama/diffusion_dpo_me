import datasets
from datasets import Dataset
import sys
import random
import mysql.connector
sys.path.append("/home/zhangyueyuan/workspace/repository/aitv_recommandation/")
from adapter.adapter.userActionDataset import UserActionDataset_v2

class MyIteratorDataset_lazy(datasets.GeneratorBasedBuilder):
    UaD = UserActionDataset_v2()
    UaD.load_dataset("/mnt/tos_zyy1/user_viewed_image/dataset_v4.h5")
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'user_id': datasets.Value('string'),
                'index': datasets.Value('int64')
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'iterator': self._get_iterator()},
            ),
        ]

    def _get_iterator(self):
        # Replace this with your actual iterator
        def iterator():
            for user_id , index in self.UaD.get_pos_neg_pair_lazy():
                yield (user_id, int(index))
                # caption = one_dict["prompts"]
                # image_data = one_dict["image_raw"]
                # yield (image_0,image_1, label, caption_0,caption_1)

        return iterator()

    def _generate_examples(self, iterator):
        for idx, (user_id, index) in enumerate(iterator):
            yield idx, {'user_id': user_id, 'index': index}

# class MyTransformed_dataset(Dataset):
#     def get_prompt_from_file_name(self, filename):
#         filename = filename.split('/')[0] + '/' + ''.join(filename.split('/')[1:])
#         config = {
#             'user': 'aitv',
#             'password': 'jaSFgt612342.!',  # 替换为你的数据库密码
#             'host': 'sh-cdb-jp89jntu.sql.tencentcdb.com',
#             'port': 63418,
#             'database': 'aitv',
#             'charset': 'utf8mb4'
#         }
#         conn = mysql.connector.connect(**config)
#         # print("数据库连接成功")
#         cursor = conn.cursor(dictionary=True) 
#         query = f"SELECT prompt FROM recommend_image_personal where image_path = %s"
#         data_to_select = (filename,)
#         cursor.execute(query,data_to_select)
#         results = cursor.fetchall()
#         # for row in results:
#         return results[0]["prompt"]

#     def transform_id_data(self,sample):
#         """
#         inputs: user_id, index
#         ouptputs: caption, jpg_0, jpg_1, label_0, label_1        
#         """
#         print(sample)
        
#         user_id_list = sample["user_id"]
#         index_list = sample["index"]

#         def transform_single(user_id, index):
#         # print(len(user_id))
#         # print(len(index))
#             user_meta = self.UaD.user_action_sequence_dict[user_id]
#             image_raw_pos = user_meta["image_raw"][index].tobytes()
#             caption = user_meta["prompt"][index].decode("utf-8")
#             neg_choice_info =  self.UaD.load_neg_pre(user_id)
#             source_id = user_meta["source_id"][index]
#             neg_choice = random.choice(neg_choice_info[str(source_id)])
#             print(neg_choice.keys())
#             neg_file_name = neg_choice["d_filename"]
#             neg_image_path = neg_choice["image_path"]
#             with open(neg_file_name, "rb") as f:
#                 body_data = f.read()
#             image_raw_neg = body_data
#             return image_raw_pos, image_raw_neg, 0, 1, caption, self.get_prompt_from_file_name(neg_image_path)
#         results = {
#             "jpg_0" :[],
#             "jpg_1" :[],
#             "caption":[],
#             "caption_neg":[],
#             "label_0":[],
#             "label_1":[]
#         }
#         for i in range(len(user_id_list)):
#             result = transform_single(user_id_list[i], index_list[i])
#             results["jpg_0"].append(result[0])
#             results["jpg_1"].append(result[1])
#             results["label_0"].append(result[2])
#             results["label_1"].append(result[3])
#             results["caption"].append(result[4])
#             results["caption_neg"].append(result[5])
#         return results

