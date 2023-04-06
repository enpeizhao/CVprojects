#include <gst/gst.h>
#include <glib.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include "nvds_yml_parser.h"

#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
#include <gst/rtsp-server/rtsp-server.h>
#include "task/border_cross.h"
#include "task/gather.h"
#include <sstream>
#include <fstream>
// gie 配置文件
#define PGIE_CONFIG_FILE "config/pgie_config.txt"
#define MAX_DISPLAY_LEN 64
// tracking 配置文件
#define TRACKER_CONFIG_FILE "config/tracker_config.txt"
#define MAX_TRACKING_ID_LEN 16

#define PGIE_CLASS_ID_VEHICLE 2
#define PGIE_CLASS_ID_PERSON 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;

Polygon g_ploygon;

// 从'./config/polygon.txt'文件中读取多边形的顶点坐标
void readPoints(std::string filename, Polygon &g_ploygon, int width, int height)
{
    std::ifstream file(filename);
    std::string str;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::string x, y;
        std::getline(ss, x, ',');
        std::getline(ss, y, ',');

        // recover to original size
        x = std::to_string(std::stof(x) * width);
        y = std::to_string(std::stof(y) * height);

        g_ploygon.push_back({std::stoi(x), std::stoi(y)});
    }
}

bool is_aarch64()
{
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

std::vector<guint> g_person_ids;
std::vector<guint> g_vehicle_ids;

typedef struct
{
    guint64 n_frames;
    guint64 last_fps_update_time;
    gdouble fps;
} PERF_DATA;

PERF_DATA g_perf_data = {0, 0, 0.0};

// 打印FPS
gboolean perf_print_callback(gpointer user_data)
{
    PERF_DATA *perf_data = (PERF_DATA *)user_data;
    guint64 current_time = g_get_monotonic_time();
    guint64 time_elapsed = current_time - perf_data->last_fps_update_time;

    if (time_elapsed > 0)
    {
        perf_data->fps = 1000000.0 * perf_data->n_frames / time_elapsed;
        g_print("FPS: %0.2f\n", perf_data->fps);
        perf_data->n_frames = 0;
        perf_data->last_fps_update_time = current_time;
    }

    return G_SOURCE_CONTINUE;
}
void update_frame_counter()
{
    g_perf_data.n_frames++;
}

/* This is the buffer probe function that we have registered on the sink pad
 * of the OSD element. All the infer elements in the pipeline shall attach
 * their metadata to the GstBuffer, here we will iterate & process the metadata
 * forex: class ids to strings, counting of class_id objects etc.
 *
 * 这是我们在OSD元素的接收器上注册的缓冲区探针函数。所有管道中的推理元素都将将其元数据附加到GstBuffer上，这里我们将迭代并处理元数据
 * 例如：类ID到字符串的转换，类ID对象的计数等。
 *
 * */

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                                   gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf); // 获取批处理元数据
    // 遍历批处理元数据，得到每一帧的元数据
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        // 获取每一帧的元数据
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int offset = 0;

        if (g_ploygon.size() == 0)
        {
            
            guint width = frame_meta->pipeline_width;
            guint height = frame_meta->pipeline_height;
            // 从配置文件恢复多边形区域
            readPoints("./config/polygon_1.txt", g_ploygon, width, height);
            g_print("read polygon.txt success!, frame height = %d, width = %d \r \n",  height, width);
        }
        // 遍历每一帧的元数据，得到每一个检测到的物体的元数据
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            // 获取每一个检测到的物体的元数据
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            // 如果是车辆
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
            {
                // 如果object id不在g_vehicle_ids中，添加进去
                if (std::find(g_vehicle_ids.begin(), g_vehicle_ids.end(), obj_meta->object_id) == g_vehicle_ids.end())
                {
                    g_vehicle_ids.push_back(obj_meta->object_id);
                }
                vehicle_count++;
                num_rects++;
            }
            // 如果是人
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
            {
                // 如果object id不在g_person_ids中，添加进去
                if (std::find(g_person_ids.begin(), g_person_ids.end(), obj_meta->object_id) == g_person_ids.end())
                {
                    g_person_ids.push_back(obj_meta->object_id);
                }
                person_count++;
                num_rects++;
            }
            // 获取检测框的中心点
            Point p = {
                obj_meta->rect_params.left + obj_meta->rect_params.width / 2,
                obj_meta->rect_params.top + obj_meta->rect_params.height / 2};
            //  如果中心点在多边形内
            if (isInside(g_ploygon, p))
            {
                // 更改检测框的颜色为红色
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 0.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                // 设置检测框的背景颜色为红色， 透明度为0.2
                obj_meta->rect_params.has_bg_color = 1;
                obj_meta->rect_params.bg_color.red = 1.0;
                obj_meta->rect_params.bg_color.green = 0.0;
                obj_meta->rect_params.bg_color.blue = 0.0;
                obj_meta->rect_params.bg_color.alpha = 0.2;
            }
            else
            {
                // 更改检测框的颜色为绿色
                obj_meta->rect_params.border_color.red = 0.0;
                obj_meta->rect_params.border_color.green = 1.0;
                obj_meta->rect_params.border_color.blue = 0.0;
            }
        }
        // 获取显示元数据，用于在屏幕上绘制多边形
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

        // 绘制多边形
        guint line_num = g_ploygon.size();
        display_meta->num_lines = line_num;

        for (guint i = 0; i < line_num; i++)
        {
            NvOSD_LineParams *line_params = &display_meta->line_params[i];
            line_params->x1 = g_ploygon[i % line_num].x;
            line_params->y1 = g_ploygon[i].y;
            line_params->x2 = g_ploygon[(i + 1) % line_num].x;
            line_params->y2 = g_ploygon[(i + 1) % line_num].y;

            line_params->line_width = 2;
            line_params->line_color.red = 1.0;
            line_params->line_color.green = 0.0;
            line_params->line_color.blue = 1.0;
            line_params->line_color.alpha = 1.0;
        }
        // 添加文字
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person Count = %d ", g_person_ids.size());
        offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "Vehicle Count = %d ", g_vehicle_ids.size());

        // 设置文字的位置
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        // 字体
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 20;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        // 背景颜色
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        // 添加显示
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

#if 0
    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
#endif
    frame_number++;
    update_frame_counter();
    return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
                   GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

/* Tracker config parsing */

#define CHECK_ERROR(error)                                                   \
    if (error)                                                               \
    {                                                                        \
        g_printerr("Error while parsing config file: %s\n", error->message); \
        goto done;                                                           \
    }

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"

static gchar *
get_absolute_file_path(gchar *cfg_file_path, gchar *file_path)
{
    gchar abs_cfg_path[PATH_MAX + 1];
    gchar *abs_file_path;
    gchar *delim;

    if (file_path && file_path[0] == '/')
    {
        return file_path;
    }

    if (!realpath(cfg_file_path, abs_cfg_path))
    {
        g_free(file_path);
        return NULL;
    }

    // Return absolute path of config file if file_path is NULL.
    if (!file_path)
    {
        abs_file_path = g_strdup(abs_cfg_path);
        return abs_file_path;
    }

    delim = g_strrstr(abs_cfg_path, "/");
    *(delim + 1) = '\0';

    abs_file_path = g_strconcat(abs_cfg_path, file_path, NULL);
    g_free(file_path);

    return abs_file_path;
}
// 从配置文件中读取配置信息， 设置tracker的属性
static gboolean set_tracker_properties(GstElement *nvtracker)
{
    gboolean ret = FALSE;
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    GKeyFile *key_file = g_key_file_new();

    if (!g_key_file_load_from_file(key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
                                   &error))
    {
        g_printerr("Failed to load config file: %s\n", error->message);
        return FALSE;
    }

    keys = g_key_file_get_keys(key_file, CONFIG_GROUP_TRACKER, NULL, &error);
    CHECK_ERROR(error);

    for (key = keys; *key; key++)
    {
        if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_WIDTH))
        {
            gint width =
                g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                       CONFIG_GROUP_TRACKER_WIDTH, &error);
            CHECK_ERROR(error);
            g_object_set(G_OBJECT(nvtracker), "tracker-width", width, NULL);
        }
        else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_HEIGHT))
        {
            gint height =
                g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                       CONFIG_GROUP_TRACKER_HEIGHT, &error);
            CHECK_ERROR(error);
            g_object_set(G_OBJECT(nvtracker), "tracker-height", height, NULL);
        }
        else if (!g_strcmp0(*key, CONFIG_GPU_ID))
        {
            guint gpu_id =
                g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                       CONFIG_GPU_ID, &error);
            CHECK_ERROR(error);
            g_object_set(G_OBJECT(nvtracker), "gpu_id", gpu_id, NULL);
        }
        else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE))
        {
            char *ll_config_file = get_absolute_file_path(TRACKER_CONFIG_FILE,
                                                          g_key_file_get_string(key_file,
                                                                                CONFIG_GROUP_TRACKER,
                                                                                CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
            CHECK_ERROR(error);
            g_object_set(G_OBJECT(nvtracker), "ll-config-file", ll_config_file, NULL);
        }
        else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE))
        {
            char *ll_lib_file = get_absolute_file_path(TRACKER_CONFIG_FILE,
                                                       g_key_file_get_string(key_file,
                                                                             CONFIG_GROUP_TRACKER,
                                                                             CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
            CHECK_ERROR(error);
            g_object_set(G_OBJECT(nvtracker), "ll-lib-file", ll_lib_file, NULL);
        }
        else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS))
        {
            gboolean enable_batch_process =
                g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                       CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
            CHECK_ERROR(error);
            g_object_set(G_OBJECT(nvtracker), "enable_batch_process",
                         enable_batch_process, NULL);
        }
        else
        {
            g_printerr("Unknown key '%s' for group [%s]", *key,
                       CONFIG_GROUP_TRACKER);
        }
    }

    ret = TRUE;
done:
    if (error)
    {
        g_error_free(error);
    }
    if (keys)
    {
        g_strfreev(keys);
    }
    if (!ret)
    {
        g_printerr("%s failed", __func__);
    }
    return ret;
}

// This function will be called when there is a new pad to be connected
static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
    g_print("In cb_newpad\n");

    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
    GstStructure *gststruct = gst_caps_get_structure(caps, 0);
    const gchar *gstname = gst_structure_get_name(gststruct);
    GstElement *source_bin = (GstElement *)data;
    GstCapsFeatures *features = gst_caps_get_features(caps, 0);

    g_print("gstname=%s\n", gstname);

    // Need to check if the pad created by the decodebin is for video and not audio.
    if (strstr(gstname, "video") != NULL)
    {
        // Link the decodebin pad only if decodebin has picked the NVIDIA
        // decoder plugin nvdec_*. We do this by checking if the pad caps contain
        // NVMM memory features.
        g_print("features=%s\n", gst_caps_features_to_string(features));

        if (gst_caps_features_contains(features, "memory:NVMM"))
        {
            // Get the source bin ghost pad
            GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");

            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), decoder_src_pad))
            {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            }

            gst_object_unref(bin_ghost_pad);
        }
        else
        {
            g_printerr("Error: Decodebin did not pick NVIDIA decoder plugin.\n");
        }
    }

    gst_caps_unref(caps);
}

static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data)
{
    g_print("Decodebin child added: %s\n", name);

    if (strstr(name, "decodebin") != NULL)
    {
        g_signal_connect(object, "child-added", G_CALLBACK(decodebin_child_added), user_data);
    }
}

// 读取视频文件
GstElement *create_source_bin(guint index, const gchar *uri)
{
    g_print("Creating source bin\n");

    // Create a source GstBin to abstract this bin's content from the rest of the pipeline
    gchar bin_name[16];
    g_snprintf(bin_name, sizeof(bin_name), "source-bin-%02d", index);
    g_print("%s\n", bin_name);
    GstElement *nbin = gst_bin_new(bin_name);

    // Source element for reading from the URI
    GstElement *uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

    // Set the input URI to the source element
    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

    // Connect to the "pad-added" signal of the decodebin
    g_signal_connect(uri_decode_bin, "pad-added", G_CALLBACK(cb_newpad), nbin);
    g_signal_connect(uri_decode_bin, "child-added", G_CALLBACK(decodebin_child_added), nbin);

    // Add the URI decode bin to the source bin
    gst_bin_add(GST_BIN(nbin), uri_decode_bin);

    // Create a ghost pad for the source bin
    GstPad *bin_pad = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
    if (!bin_pad)
    {
        g_printerr("Failed to add ghost pad in source bin\n");
        return NULL;
    }

    gst_element_add_pad(nbin, bin_pad);
    return nbin;
}

// main函数
int main(int argc, char *argv[])
{

    GMainLoop *loop = NULL;
    // 创建各种元素
    GstElement *pipeline = NULL, *source = NULL, *streammux = NULL, *pgie = NULL, *nvvidconv = NULL,
               *nvosd = NULL, *nvtracker = NULL, *nvvidconv_postosd = NULL, *caps = NULL, *encoder = NULL, *rtppay = NULL, *sink = NULL;
    g_print("With tracker\n");
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    GstPad *osd_sink_pad = NULL;
    GstCaps *caps_filter = NULL;

    guint bitrate = 5000000;       // 比特率
    gchar *codec = "H264";         // 设置编码格式
    guint updsink_port_num = 5400; // 设置端口号
    guint rtsp_port_num = 8554;    // 设置RTSP端口号
    gchar *rtsp_path = "/ds-test"; // 设置RTSP路径

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    /* Check input arguments */
    if (argc != 2)
    {
        g_printerr("OR: %s <H264 filename>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    // 初始化GStreamer
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    // ==================== 创建元素 ====================

    pipeline = gst_pipeline_new("ds-tracker-pipeline"); // 创建管道

    source = create_source_bin(0, argv[1]);                                              // 创建source_bin元素， 用于从文件中读取视频流
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");                 // 创建流复用器， 用于将多个流合并为一个流 ， 以及将多帧画面打包batch
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");            // 创建PGIE元素， 用于执行推理
    nvtracker = gst_element_factory_make("nvtracker", "tracker");                        // 创建tracker元素， 用于跟踪识别到的物体
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");         // 创建nvvidconv元素， 用于将NV12转换为RGBA
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");                   // 创建nvosd元素， 用于在转换后的RGBA缓冲区上绘制
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor_postosd"); // 创建nvvidconv_postosd元素， 用于将NV12转换为RGBA
    caps = gst_element_factory_make("capsfilter", "filter");                             // 创建caps元素， 用于设置视频格式

    // 创建编码器
    if (g_strcmp0(codec, "H264") == 0)
    {
        // 创建H264编码器
        encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
        printf("Creating H264 Encoder\n");
    }
    else if (g_strcmp0(codec, "H265") == 0)
    {
        // 创建H265编码器
        encoder = gst_element_factory_make("nvv4l2h265enc", "encoder");
        printf("Creating H265 Encoder\n");
    }

    //  创建rtppay元素， 用于将编码后的数据打包为RTP包
    if (g_strcmp0(codec, "H264") == 0)
    {
        rtppay = gst_element_factory_make("rtph264pay", "rtppay");
        printf("Creating H264 rtppay\n");
    }
    else if (g_strcmp0(codec, "H265") == 0)
    {
        rtppay = gst_element_factory_make("rtph265pay", "rtppay");
        printf("Creating H265 rtppay\n");
    }
    // 创建udpsink元素， 用于将RTP包发送到网络
    sink = gst_element_factory_make("udpsink", "udpsink");

    if (!source || !pgie || !nvtracker || !nvvidconv || !nvosd || !nvvidconv_postosd ||
        !caps || !encoder || !rtppay || !sink)
    {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    // ==================== 设置元素参数 ====================

    // 1.设置streammux元素的参数
    g_object_set(G_OBJECT(streammux), "batch-size", 1, NULL);
    g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                 MUXER_OUTPUT_HEIGHT,
                 "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    // 2.设置PGIE元素的参数
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);

    // 3.设置tracker元素的参数
    set_tracker_properties(nvtracker);

    // 4.设置caps元素的视频格式
    caps_filter = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
    g_object_set(G_OBJECT(caps), "caps", caps_filter, NULL);
    // 释放caps_filter
    gst_caps_unref(caps_filter);

    // 5.设置编码器的比特率
    g_object_set(G_OBJECT(encoder), "bitrate", bitrate, NULL);
    // 设置编码器的preset-level
    if (is_aarch64())
    {
        g_object_set(G_OBJECT(encoder), "preset-level", 1, NULL);
        g_object_set(G_OBJECT(encoder), "insert-sps-pps", 1, NULL);
    }

    // 6.设置udpsink元素的参数
    g_object_set(G_OBJECT(sink), "host", "224.224.255.255", NULL);
    g_object_set(G_OBJECT(sink), "port", updsink_port_num, NULL);
    g_object_set(G_OBJECT(sink), "async", FALSE, NULL);
    g_object_set(G_OBJECT(sink), "sync", 1, NULL);

    // 添加消息处理器
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // ==================== 将元素添加到管道中 ====================
    gst_bin_add_many(GST_BIN(pipeline),
                     source, streammux, pgie, nvtracker,
                     nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL);

    // ==================== 将source_bin 添加到streammux元素的sink pad ====================
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    // 获取streammux元素的sink pad
    sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
    // 获取source_bin元素的src pad
    srcpad = gst_element_get_static_pad(source, pad_name_src);

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }
    // 释放sinkpad和srcpad
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    //  ==================== 将元素链接起来 ====================
    if (!gst_element_link_many(streammux, pgie, nvtracker,
                               nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL))
    {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    // 添加探针，用于获取元数据
    osd_sink_pad = gst_element_get_static_pad(nvtracker, "src"); // 获取nvosd元素的sink pad
    if (!osd_sink_pad)
        g_print("Unable to get sink pad\n");
    else
        // 参数：pad, 探针类型, 探针回调函数, 回调函数的参数, 回调函数的参数释放函数
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL); // 添加探针
    g_timeout_add(5000, perf_print_callback, &g_perf_data);                                                // 添加定时器，用于打印性能数据
    gst_object_unref(osd_sink_pad);

    // ==================== 创建rtsp服务器， 用于将视频流发布到网络 ====================
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;

    server = gst_rtsp_server_new();
    g_object_set(G_OBJECT(server), "service", g_strdup_printf("%d", rtsp_port_num), NULL);
    gst_rtsp_server_attach(server, NULL);
    mounts = gst_rtsp_server_get_mount_points(server);

    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, g_strdup_printf("( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )", updsink_port_num, codec));
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, rtsp_path, factory);

    g_object_unref(mounts);

    printf("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d%s ***\n\n", rtsp_port_num, rtsp_path);

    // ==================== 启动管道 ====================
    /* Set the pipeline to "playing" state */
    g_print("Using file: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Iterate */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL); // 设置管道状态为NULL
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
