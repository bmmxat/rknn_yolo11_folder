// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"


#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <model_path> <image_folder> [options]\n", argv[0]);
        printf("Options:\n");
        printf("  --conf <float>     设置置信度阈值 (默认: 0.3)\n");
        printf("  --nms <float>      设置NMS阈值 (默认: 0.5)\n");
        printf("  --output <path>    设置检测结果图片输出目录\n");
        printf("  --save-txt         保存检测结果到txt文件\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_folder = argv[2];

    float conf_threshold = 0.3;
    float nms_threshold = 0.5;
    const char *output_dir = nullptr;
    int save_txt = 0; // 新增：控制是否保存txt

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
            conf_threshold = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--nms") == 0 && i + 1 < argc) {
            nms_threshold = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[i + 1];
            if (access(output_dir, F_OK) != 0) {
                if (mkdir(output_dir, 0755) != 0) {
                    printf("Failed to create output directory: %s\n", output_dir);
                    return -1;
                }
            }
            i++;
        } else if (strcmp(argv[i], "--save-txt") == 0) { // 新增参数
            save_txt = 1;
        }
    }

    int ret;
    rknn_app_context_t rknn_app_ctx;
    DIR *dir = NULL;
    double total_inference_time = 0.0;
    int image_count = 0;
    int total_files = 0;

    init_post_process();
    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        return ret;
    }

    dir = opendir(image_folder);
    if (!dir)
    {
        printf("Failed to open directory: %s\n", image_folder);
        goto out;
    }

    struct dirent *entry;
    struct stat file_stat;

    while ((entry = readdir(dir)) != NULL)
    {
        #define MAX_PATH_LEN 1024
        char file_path[MAX_PATH_LEN];
        size_t path_len = snprintf(file_path, sizeof(file_path), "%s/%s", image_folder, entry->d_name);
        if (path_len >= sizeof(file_path)) {
            printf("Error: Path too long\n");
            continue;
        }

        if (stat(file_path, &file_stat) == 0 && S_ISREG(file_stat.st_mode))
        {
            total_files++;
        }
    }

    rewinddir(dir);

    while ((entry = readdir(dir)) != NULL)
    {
        #define MAX_PATH_LEN 1024
        char file_path[MAX_PATH_LEN];
        size_t path_len = snprintf(file_path, sizeof(file_path), "%s/%s", image_folder, entry->d_name);
        if (path_len >= sizeof(file_path)) {
            printf("Error: Path too long\n");
            continue;
        }

        // 检查是否为文件
        if (stat(file_path, &file_stat) == 0 && S_ISREG(file_stat.st_mode))
        {
            image_buffer_t src_image;
            memset(&src_image, 0, sizeof(image_buffer_t));

            ret = read_image(file_path, &src_image);
            if (ret != 0)
            {
                printf("Failed to read image: %s\n", file_path);
                continue;
            }
            struct timespec start_time, end_time;
            clock_gettime(CLOCK_MONOTONIC, &start_time);

            object_detect_result_list od_results;
            printf("***debug***  --detect start\n");
            ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results, conf_threshold, nms_threshold);
            printf("***debug***  --detect finish\n");
            clock_gettime(CLOCK_MONOTONIC, &end_time);

            if (ret != 0)
            {
                printf("Inference failed for image: %s\n", file_path);
                continue;
            }

            double inference_time = (end_time.tv_sec - start_time.tv_sec) +
                                    (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
            total_inference_time += inference_time;
            image_count++;

            printf("Image: %s, Inference time: %.3f seconds\n", file_path, inference_time);

            printf("\rProcessing: %d/%d images (%.1f%%)", 
                   image_count, total_files, 
                   (float)image_count / total_files * 100);
            fflush(stdout);

            if (output_dir != nullptr) {
                char text[256];
                for (int i = 0; i < od_results.count; i++)
                {
                    object_detect_result *det_result = &(od_results.results[i]);
                    printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                           det_result->box.left, det_result->box.top,
                           det_result->box.right, det_result->box.bottom,
                           det_result->prop);
                    int x1 = det_result->box.left;
                    int y1 = det_result->box.top;
                    int x2 = det_result->box.right;
                    int y2 = det_result->box.bottom;

                    draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

                    sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
                    draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
                }

                char output_path[1024];
                const char *input_filename = strrchr(file_path, '/');
                input_filename = input_filename ? input_filename + 1 : file_path;
                snprintf(output_path, sizeof(output_path), "%s/det_%s", output_dir, input_filename);
                write_image(output_path, &src_image);

                // 新增：保存检测结果到txt
                if (save_txt) {
                    char txt_path[1024];
                    snprintf(txt_path, sizeof(txt_path), "%s/det_%s", output_dir, input_filename);
                    char *dot = strrchr(txt_path, '.');
                    if (dot) strcpy(dot, ".txt");
                    FILE *fp = fopen(txt_path, "w");
                    if (fp) {
                        fprintf(fp, "ID,PATH,TYPE,SCORE,XMIN,YMIN,XMAX,YMAX\n");
                        for (int i = 0; i < od_results.count; i++) {
                            object_detect_result *det_result = &(od_results.results[i]);
                            fprintf(fp, "%d,%s,%s,%.3f,%d,%d,%d,%d\n",
                                i,
                                file_path,
                                coco_cls_to_name(det_result->cls_id),
                                det_result->prop,
                                det_result->box.left,
                                det_result->box.top,
                                det_result->box.right,
                                det_result->box.bottom
                            );
                        }
                        fclose(fp);
                    } else {
                        printf("Failed to write result txt: %s\n", txt_path);
                    }
                }
            }

#if defined(RV1106_1103)
            dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                         rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
            free(src_image.virt_addr);
#endif
        }
    }

    closedir(dir);

    if (image_count > 0)
    {
        printf("Processed %d images, Average inference time: %.3f seconds\n",
               image_count, total_inference_time / image_count);
    }
    else
    {
        printf("No valid images found in the folder.\n");
    }

out:
    deinit_post_process();

    ret = release_yolo11_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolo11_model fail! ret=%d\n", ret);
    }

    return 0;
}

