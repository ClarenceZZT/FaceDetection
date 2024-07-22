import matplotlib.pyplot as plt
import numpy as np
def draw_keypoints(image, outputs, gt):
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    # print(outputs.shape,gt.shape)
    for i in range(64):
        img = image[i,:,:]
        output_keypoint = outputs[i]
        ground_truth = gt[i]
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))
        
        plt.imshow(img)
        output_keypoint = output_keypoint.reshape(-1, 2)
        for p in range(output_keypoint.shape[0]):
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')

        ground_truth =ground_truth.reshape(-1, 2)
        for p in range(ground_truth.shape[0]):
            plt.plot(ground_truth[p, 0], ground_truth[p, 1], 'b.') 

        plt.savefig(f"../outputs/test/val_{i}.png")
        plt.close()

def show_keypoint_location(data):
    #
    plt.figure(figsize=(90, 90))
    for i in range(4):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], 'o', color='red',markersize=20)
    plt.show()
    plt.close()

def calculate_delta(points):
    mouth_width = abs(points[54][0] - points[48][0])
    mouth_height = abs(points[57][1] - points[51][1])

    delta = (mouth_width + mouth_height) / 30
    return delta

def judge_expression_weighted(points, weights):
    delta = calculate_delta(points)
    weighted_sum = sum(point[1] * weight for point, weight in zip(points, weights))
    total_weight = sum(weights)
    weighted_average_y = weighted_sum / total_weight
    
    upper_lip_avg = (points[50][1] + points[51][1] + points[52][1]) / 3
    lower_lip_avg = (points[58][1] + points[57][1] + points[56][1]) / 3
    reference_line = (upper_lip_avg + lower_lip_avg) / 2

    print(delta,weighted_average_y - reference_line)
    if weighted_average_y - reference_line < 0.5*delta and weighted_average_y-reference_line>-delta:
        return "Neutral"
    elif weighted_average_y < reference_line:
        return "Positive"
    else:
        return "Negative"