class IOU():
  """
  Aim of this class is to compute Intersection over Union score based on given points of bounding boxes.
  """
  def __init__(self):
    pass

  def area_of_points(self,tensor):
    """
    This function calculates area of a box.
    :param tensor: points of bounding box.
    :return: area of bounding box.
    """
    return torch.abs(torch.prod(tensor,dim=1))
  
  def cal_mid_point(self,right_down_corner,left_up_corner):
    """
    This function computes middle point between given two points.
    :param right_down_corner: bounding box's right and down point.
    :param left_up_corner: bounding box's left and up point.
    :return: middle point
    """
    diff_cor = (right_down_corner - left_up_corner) / 2
    mid_point = diff_cor + left_up_corner
    return mid_point
  
  def check_the_Point_in_box(self,point, query_box):
    """
    The aim of this function is to decide whether given point is in given query box.
    :param point: Coordinates of one point
    :param query_box: Coordinates of a bounding box
    :return:
    """
    condition_1 = torch.where(query_box[:,0] < point[:,0],True, False)
    condition_2 = torch.where(point[:,0] < query_box[:,2],True, False)
    condition_3 = torch.where(query_box[:,1] < point[:,1],True, False)
    condition_4 = torch.where(point[:,1] < query_box[:,3],True, False)
    return condition_1 * condition_2 * condition_3 * condition_4
  
  def check_intersection(self,right_down_corner,left_up_corner,pred_box,gt_box):
    """
    The purpose of this function is to decide whether there is an intersection between boxes or not.
    :param right_down_corner: calculated intersection box's right and down point
    :param left_up_corner: calculated intersection box's left and up point
    :param pred_box: points of predicted boxes.
    :param gt_box: points of ground truth boxes.
    :return:
    """
    mid_point = self.cal_mid_point(right_down_corner,left_up_corner)
    state_pre = self.check_the_Point_in_box(mid_point,pred_box)
    state_gt = self.check_the_Point_in_box(mid_point, gt_box)
    return state_pre * state_gt
  
  def __call__(self,pred_box,gt_box):
    """
    This function computes Intersection over Union score between given boxes.
    Intersection over Union score is calculated based on formula below:
    IoU = Area of Intersection / (Area of predicted bounding box + Area of ground truth bounding box - Area of
    Intersection)
    :param pred_box: points of predicted boxes.
    :param gt_box: points of ground truth boxes.
    :return: Intersection over Union score
    """
    intersection_left_up_corner = torch.max(pred_box[:,0:2],gt_box[:,0:2])
    intersection_right_down_corner = torch.min(pred_box[:,2:],gt_box[:,2:])

    intersection_area = self.area_of_points((intersection_right_down_corner - intersection_left_up_corner))
    condition = self.check_intersection(intersection_right_down_corner, intersection_left_up_corner,pred_box,gt_box)
    area_of_intersection = torch.where(condition,intersection_area,torch.tensor(0.0).to(device))

    area_of_pred_box = self.area_of_points(pred_box[:,0:2]-pred_box[:,2:])
    area_of_gt_box = self.area_of_points(gt_box[:,0:2]-gt_box[:,2:])

    IoU = area_of_intersection / (area_of_pred_box + area_of_gt_box- area_of_intersection)
    return IoU.to(device)

class Avr_Pre_Recall():
    """
    Aim of this class is to calculate Average Precision and Average Recall based on given IoU threshold.
    To achieve that goal it should collect True Positive, False Positive, probabilities of predicting bounding boxes
    and number of ground truth bounding boxes for each class.
    """
    def __init__(self):
        self.Iou_func = IOU()
        self.aver_pred_recall = {}

    def calculate_TP_FP(self,predictions,score_list, GT, TP, FP, threshold):
        """
        This function simply computes True Positive and False Positive for given predicted bounding boxes and
        ground truth bounding boxes.
        True Positive: when there is a predicted bounding boxes which is not matched any ground-truth box before and
        its IoU score is greater than the threshold.
        False Positive: when there is a predicted bounding boxes which is matched a ground-truth box before or
        its IoU score is less than the threshold.
        False Negative: when there is a ground-truth bounding box but the model did not generate any predicted bounding
        box for it.
        :param predictions: predicted bounding boxes
        :param score_list: probability of predicted bounding boxes
        :param GT: ground thruth bounding boxes
        :param TP: True Positive list
        :param FP: False Positive list. Length of this list is equal to number of prediction bounding boxes.
        :param threshold: IoU threshold value. It is used to consider whether bounding box match with ground truth
        bounding box
        :return: True Positive list, False Positive list, score_list
        """
        used_pred_idx = []
        for gt_idx, gt in enumerate(GT):
            list_of_IoU_gt = []

            index_order = torch.argsort(score_list,descending=True)
            predictions = predictions[index_order]
            score_list = score_list[index_order]

            #rather than using another "for loop", gt bounding box is copied and the matrix operations are used.
            multiple_gt = gt.repeat(predictions.size(0),1)
            list_of_IoU_gt = self.Iou_func(predictions, multiple_gt)

            max_index = torch.argmax(list_of_IoU_gt)
            if list_of_IoU_gt[max_index] > threshold and max_index not in used_pred_idx:
              TP[max_index] = 1
              FP[max_index] = 0
              used_pred_idx.append(max_index)


        return TP, FP, score_list

    def creating_conf_dic(self, predictions, gt_info, class_number, conf_dic, threshold):
        """
        This function collects calculated True Positive, False Positive, probabilities of predicting bounding boxes
        and number of ground truth bounding boxes in conf_dic.
        :param predictions: predicted bounding boxes
        :param gt_info: ground truth bounding boxes
        :param class_number: total number of class
        :param threshold: Intersection over Union threshold
        """
        for cls_idx in range(class_number):
            #Selecting bounding boxes whose class number matchs with current class index
            index_of_cls = (predictions['labels'] == cls_idx).nonzero(as_tuple=True)[0]
            index_of_gt_cls = (gt_info['labels'] == cls_idx).nonzero(as_tuple=True)[0]

            number_of_pred_box = len(predictions['boxes'][index_of_cls])
            number_of_gt_box = len(gt_info['boxes'][index_of_gt_cls])
            TP, FP = torch.zeros(number_of_pred_box), torch.ones(number_of_pred_box)

            score_list = predictions['scores'][index_of_cls]

            number_of_GT = len(index_of_gt_cls)

            if number_of_gt_box > 0 and number_of_pred_box > 0:
                "Calculating True Positive and False Positive"
                TP, FP, score_list = self.calculate_TP_FP(predictions['boxes'][index_of_cls],score_list,
                                                                                   gt_info['boxes'][index_of_gt_cls],TP,FP,
                                                                                   threshold=threshold)
            TP = TP.tolist()
            FP = FP.tolist()
            score_list = score_list.tolist()
            if cls_idx not in conf_dic.keys():
                value_of_conf = {"TP":TP, "FP":FP, "number_of_GT":number_of_GT, "score_list":score_list}
                conf_dic.update({cls_idx: value_of_conf})
            else:
                conf_dic[cls_idx]["TP"] += TP
                conf_dic[cls_idx]["FP"] += FP
                conf_dic[cls_idx]["number_of_GT"] += number_of_GT
                conf_dic[cls_idx]["score_list"] += score_list

    def check_inputs_are_Tensor(self,input_1,input_2):
        """
        This function is used to make sure these two inputs are Tensor.
        """
        if not isinstance(input_1,torch.Tensor):
            input_1 = torch.Tensor(input_1)
        if not isinstance(input_2,torch.Tensor):
            input_2 = torch.Tensor(input_2)
        return input_1, input_2

    def calculate_metrics(self,TP, FP,number_of_GT,return_tensor=True,epsilon=10**(-6)):
        """
        The goal of this function is to compute precision and recall for given values.
        :param TP_cumulative_sum: cumulative sum of True Positive
        :param FP: cumulative sum of False Positive
        :param number_of_GT: total number of ground truth
        :return: precision tensor and recall tensor
        """

        TP, FP = self.check_inputs_are_Tensor(TP, FP)
        precision = TP / (TP + FP +epsilon)
        recall = TP / (number_of_GT + epsilon)
        if return_tensor:
            precision = torch.Tensor(precision)
            recall = torch.Tensor(recall)

        return precision, recall

    def calculate_area_under_curve(self,input_1,input_2):
        """
        This function was writen to calculate an area under curve
        :param input_1: x-axis of the curve
        :param input_2: y-axis of the curve
        :return: area
        """
        input_1, input_2 = self.check_inputs_are_Tensor(input_1, input_2)
        #torch.sum((input_1[1:] - input_1[:-1]) * input_2[:-1])
        return torch.trapz(input_2, input_1)

    def calculate_avr_precision_recall(self,conf_dic,threshold, class_index=1, draw_graph=True):
        """
        The aim of this function is to compute precision and recall value according to following formulas.
        precision = TP /(TP+FP), recall = TP/(TP+FN)
        :param conf_dic: dictionary which contains information about TP, FP, total number of GT boxes
        :param draw_graph: parameter to understand drawing graphs or not
        :param select_class_index: class_index to draw metric results of that class
        :return: dictionary of average precision and recall
        """


        number_of_GT = conf_dic[class_index]["number_of_GT"]

        score_list = conf_dic[class_index]["score_list"]
        index_order = np.argsort(score_list)[::-1]
        TP = np.array(conf_dic[class_index]["TP"])[index_order]
        FP = np.array(conf_dic[class_index]["FP"])[index_order]


        TP_cumulative_sum = np.cumsum(TP)
        FP_cumulative_sum = np.cumsum(FP)


        precision_tensor, recall_tensor = self.calculate_metrics(TP_cumulative_sum, FP_cumulative_sum,number_of_GT)

        AP = self.calculate_area_under_curve(recall_tensor, precision_tensor)

        TP = [np.sum(TP)]
        FP = [np.sum(FP)]
        precision, recall = self.calculate_metrics(TP, FP, number_of_GT,return_tensor=False)
        if class_index not in self.aver_pred_recall.keys():
            recall_list = [recall]
            AP_list = [AP]
            threshold_list = [threshold]
            precision_list = [precision]
        else:
            AP_list = self.aver_pred_recall[class_index][0] + [AP]
            precision_list = self.aver_pred_recall[class_index][1] + [precision]
            recall_list = self.aver_pred_recall[class_index][2] + [recall]
            threshold_list = self.aver_pred_recall[class_index][3] + [threshold]

        
        AR = self.calculate_area_under_curve(threshold_list,recall_list)
        print("Average Recall ", AR.item(), "for class_index ", class_index)

        if draw_graph:
            print("Average Precision ",AP.item(),"for class_index ",class_index,"with",threshold,"IoU threshold")
            plt.plot(np.array(recall_tensor), np.array(precision_tensor))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()

            

            if len(threshold_list) > 1:
                plt.plot(np.array(threshold_list), np.array(recall_list))
                plt.xlabel('IoU threshold')
                plt.ylabel('Recall')
                plt.show()

        self.aver_pred_recall.update({class_index:[AP_list, precision_list, recall_list, threshold_list]})
        return self.aver_pred_recall

def eval():
    # TO DO
    model.eval()
    print("device:",device)
    model.to(device)
    aver_pre_recall = Avr_Pre_Recall()
    for threshold in range(1, 10, 3):
        threshold /= 10
        print("#"*50)
        print("IoU threshold:", threshold)
        conf_dic = {}
        selected_class_index = 1
        for images, gt_info in data_loader:
            #= data
            images = images[0].to(device),
            gt_info[0]['boxes'] = gt_info[0]['boxes'].to(device)
            gt_info[0]['labels'] = gt_info[0]['labels'].to(device)


            with torch.no_grad():
                predictions = model(images)

            predictions[0]['boxes'] = predictions[0]['boxes'].to(device)
            predictions[0]['scores'] = predictions[0]['scores'].to(device)
            predictions[0]['labels'] = predictions[0]['labels'].to(device)

            aver_pre_recall.creating_conf_dic(predictions[0], gt_info[0], class_number, conf_dic, threshold)


        aver_pre_recall.calculate_avr_precision_recall(conf_dic,threshold,class_index=selected_class_index,draw_graph=True)


eval()
