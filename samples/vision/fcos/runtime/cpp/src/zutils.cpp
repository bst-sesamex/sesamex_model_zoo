#include "zutils.h"


bool equal(float a, float b) {
    return std::fabs(a - b) < EPSILON;
}

/**
 * @brief given (c, y, x) grid coordinate in DWH format tensor, get 1D index in tensor
 */
int64_t GetIndex(int64_t c, int64_t y, int64_t x, int64_t width, int64_t height, int64_t num_channels) {
    return ((y * width) + x) * num_channels + c;
}

/**
 * @brief Given 1D index in DWH tensor, return its grid coordinate
 */
std::tuple<int64_t, int64_t , int64_t> GetCYX(int64_t index, int64_t width, int64_t height, int64_t num_channels) {
    int64_t c = index % num_channels;
    int64_t xy = index / num_channels;
    int64_t y = xy / width;
    int64_t x = xy % width;

    return std::make_tuple(c, y, x);  
}


// max pooling op on DWH format tensor
void max_pool_2d(float* input_ts, 
                const int width, const int height, const int channels, 
                float* output_ts, int kernel_size, int padding, int stride) {
    
    int out_width = (width - kernel_size + 2 * padding) / stride + 1;
    int out_height = (height - kernel_size + 2 * padding) / stride + 1;

    for(int i = 0; i < channels; i++) {
        for(int j = 0; j < out_height; j++) {
            for(int k = 0; k < out_width; k++) {
                int y_origin = j * stride;
                int x_origin = k * stride;

                int y_start = y_origin - padding;
                int x_start = x_origin - padding;

                int y_end = std::min(y_start + kernel_size, height);
                int x_end = std::min(x_start + kernel_size, width);

                y_start = std::max(y_start, 0);
                x_start = std::max(x_start, 0);

                float max_val = std::numeric_limits<float>::lowest();
                for(int y = y_start; y < y_end; y++) {
                    for(int x = x_start; x < x_end; x++) {
                        int64_t index = GetIndex(i, y, x, width, height, channels);
                        if(input_ts[index] > max_val) {
                            max_val = input_ts[index];
                        }
                    }
                }

                int index_out = GetIndex(i, j, k, out_width, out_height, channels);
                output_ts[index_out] = max_val;
            }
        }
    }
}

void test_maxpool() {
    float* input = new float[36];
    for(int i = 0; i < 36; i++) {
        input[i] = i * 1.0 + 1.0;
    }

    float* output = new float[36] {0.0};
    max_pool_2d(input, 6, 6, 1, output, 3, 1, 1);
    for(int i = 0; i < 36; i++) {
        std::cout << output[i] <<"\t";
        if((i + 1) % 6 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}


void RGB2NV12(const cv::Mat& src, uint8_t * yuvbuff)
{
    int cols = src.cols;
    int rows = src.rows;

    int Y_index = 0;
    int UV_index = rows * cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int B = src.at<cv::Vec3b>(i, j)[0];
            int G = src.at<cv::Vec3b>(i, j)[1];
            int R = src.at<cv::Vec3b>(i, j)[2];
            
            int Y = (77 * R + 150 * G + 29 * B) >> 8;
            yuvbuff[Y_index++] = (Y < 0) ? 0 : ((Y > 255) ? 255 : Y);
            
            if((i % 2 == 0) && (j % 2 == 0))
            {
                int U = ((-44 * R -87 * G + 131 * B) >> 8) + 128;
                int V = ((131 * R - 110 * G - 21 * B) >> 8) + 128;
                yuvbuff[UV_index++] = (U < 0) ? 0 : ((U > 255) ? 255 : U);
                yuvbuff[UV_index++] = (V < 0) ? 0 : ((V > 255) ? 255 : V);
            }
        }
    }
}


void softmax(float* input_p, int C, int H, int W)
{
    for(int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            float sum = 0.0;
            float max = 0.0;
            for (int k = 0; k < C; k++)
            {
                if (max < input_p[k * W * H + i * W + j])
                    max = input_p[k * W * H + i * W + j];
            }

            for (int k = 0; k < C; k++)
            {
                input_p[k * W * H + i * W + j] = exp(input_p[k * W * H + i * W + j] - max);
                sum += input_p[k * W * H + i * W + j];
            }
            for(int k = 0; k < C; k++)
            {
                input_p[k * W * H + i * W + j] /= sum;  
            }
        }
    }
}

void argmax(float* input_p, uint8_t* output_p, int C, int H, int W)
{
    float max_value = 0.0;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            for(int k = 0; k < C; k++)
            {
                if(input_p[k * W * H + i * W + j] > max_value)
                {
                    max_value = input_p[k * W * H + i * W + j];
                    output_p[i * W + j] = k;
                }
            }
        }
    }
}


static float IOU(const ObjInfo& a, const ObjInfo& b)
{
  
  auto overlap1D = [](float x1min, float x1max,
		      float x2min, float x2max) -> float {
    
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };
  
  float overlap_x = overlap1D(a.x1, a.x2, b.x1, b.x2);
  float overlap_y = overlap1D(a.y1, a.y2, b.y1, b.y2);
  float area_intersection = overlap_x * overlap_y;
  
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  
  float area_union = area_a + area_b - area_intersection;
  
  return area_union == 0.f ? 0.f : area_intersection / area_union;
}

void soft_nms(std::vector<ObjInfo>& bboxes_in, std::vector<ObjInfo>& bboxes_out, 
            const int method, const float& sigma, const float& iou_thresh, const float& threshold) {
    if(bboxes_in.empty()) {
        return;
    }
    bboxes_out.clear();

    float weight = 0.0;
    // sort bboxes with scores by descending order
    std::sort(bboxes_in.begin(), bboxes_in.end(), 
            [](const ObjInfo& a, const ObjInfo& b) {
                return a.score > b.score;
            });
    for(auto i = 0; i < bboxes_in.size(); i++) {
        ObjInfo& box = bboxes_in[i];

        bool is_valid_box = true;
        for(int j = 0; j < bboxes_out.size(); j++) {
            ObjInfo& box_out = bboxes_out[j];
            float iou = IOU(box, box_out);

            if(box.classId != box_out.classId) {
                iou = 0.0;
            }

            if(method == 1) { //liear method
                if(iou > iou_thresh) {
                    weight = 1 - iou;
                } else {
                    weight = 1.0;
                }
            } else if (method == 2) { // gaussian method
                weight = exp(-iou * iou / sigma);
            } else { // origin NMS
                if(iou > iou_thresh) {
                    weight = 0.0;
                } else {
                    weight = 1.0;
                }
            }
            box.score *= weight;

            if(box.score < threshold) {
                is_valid_box = false;
                break;
            }   
        }

        if(is_valid_box) {
            bboxes_out.push_back(box);
        }

    }
}

