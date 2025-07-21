package ai.basic.x1.adapter.api.job;

import ai.basic.x1.adapter.api.job.converter.YoloV8RequestConverter;
import ai.basic.x1.adapter.api.job.converter.YoloV8ResponseConverter;
import ai.basic.x1.adapter.dto.ApiResult;
import ai.basic.x1.adapter.dto.PreModelParamDTO;
import ai.basic.x1.adapter.port.dao.mybatis.model.DataAnnotationObject;
import ai.basic.x1.adapter.port.dao.mybatis.model.ModelDatasetResult;
import ai.basic.x1.adapter.port.dao.mybatis.model.ModelRunRecord;
import ai.basic.x1.adapter.port.rpc.YoloV8ModelHttpCaller;
import ai.basic.x1.adapter.port.rpc.dto.YoloV8MetricsReqDTO;
import ai.basic.x1.adapter.port.rpc.dto.YoloV8Object;
import ai.basic.x1.adapter.port.rpc.dto.YoloV8RespDTO;
import ai.basic.x1.entity.*;
import ai.basic.x1.entity.enums.DataAnnotationObjectSourceTypeEnum;
import ai.basic.x1.entity.enums.ModelCodeEnum;
import ai.basic.x1.usecase.ModelUseCase;
import ai.basic.x1.usecase.exception.UsecaseCode;
import ai.basic.x1.usecase.exception.UsecaseException;
import ai.basic.x1.util.DefaultConverter;
import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * YOLOv8 ImageNet Model Handler
 * @author AI Assistant
 */
@Slf4j
public class YoloV8ImageNetModelHandler extends AbstractModelMessageHandler<YoloV8RespDTO> {

    @Autowired
    private YoloV8ModelHttpCaller modelHttpCaller;

    @Autowired
    private ModelUseCase modelUseCase;

    @Value("${yolov8.resultEvaluate.url:}")
    private String resultEvaluateUrl;

    @Override
    public ModelTaskInfoBO modelRun(ModelMessageBO message) {
        log.info("start YOLOv8 model run. dataId: {}, modelSerialNo: {}", message.getDataId(),
                message.getModelSerialNo());
        var apiResult = getRetryAbleApiResult(message);
        var systemModelClassMap = modelUseCase.getModelClassMapByModelId(message.getModelId());
        var filterCondition = JSONUtil.toBean(message.getResultFilterParam(),
                PreModelParamDTO.class);
        return YoloV8ResponseConverter.convert(apiResult, systemModelClassMap, filterCondition);
    }

    @Override
    ApiResult<YoloV8RespDTO> callRemoteService(ModelMessageBO message) {
        try {
            var apiResult = modelHttpCaller
                    .callYoloV8Model(YoloV8RequestConverter.convert(message), message.getUrl());

            if (CollUtil.isNotEmpty(apiResult.getData())) {
                return new ApiResult<>(apiResult.getCode(), apiResult.getMessage(),
                        apiResult.getData().get(0));
            }
            return new ApiResult<>(apiResult.getCode(), apiResult.getMessage());
        } catch (Exception e) {
            log.error("call YOLOv8 model error", e);
            throw new UsecaseException(UsecaseCode.UNKNOWN, e.getMessage());
        }
    }

    @Override
    public void syncModelAnnotationResult(ModelTaskInfoBO modelTaskInfo, ModelMessageBO modelMessage) {
        var modelResult = (YoloV8ObjectBO) modelTaskInfo;
        if (CollUtil.isNotEmpty(modelResult.getObjects())) {
            var lambdaQueryWrapper = Wrappers.lambdaQuery(ModelRunRecord.class);
            lambdaQueryWrapper.eq(ModelRunRecord::getModelSerialNo, modelMessage.getModelSerialNo());
            lambdaQueryWrapper.last("limit 1");
            var modelRunRecord = modelRunRecordDAO.getOne(lambdaQueryWrapper);
            var dataAnnotationObjectBOList = new ArrayList<DataAnnotationObjectBO>(modelResult.getObjects().size());
            modelResult.getObjects().forEach(o -> {
                var dataAnnotationObjectBO = DataAnnotationObjectBO.builder()
                        .datasetId(modelMessage.getDatasetId()).dataId(modelResult.getDataId()).classAttributes(JSONUtil.parseObj(o))
                        .sourceType(DataAnnotationObjectSourceTypeEnum.MODEL).sourceId(modelRunRecord.getId()).build();
                dataAnnotationObjectBOList.add(dataAnnotationObjectBO);
            });

            dataAnnotationObjectDAO.saveBatch(DefaultConverter.convert(dataAnnotationObjectBOList, DataAnnotationObject.class));
        }
    }

    @Override
    public void assembleCalculateMetricsData(List<ModelDatasetResult> modelDatasetResults, List<DataAnnotationObject> dataAnnotationObjectList,
                                             String groundTruthFilePath, String modelRunFilePath) {
        if (CollUtil.isEmpty(modelDatasetResults)) {
            return;
        }
        var dataAnnotationObjectMap = dataAnnotationObjectList.stream().filter(dataAnnotationObject -> {
            var objectBO = DefaultConverter.convert(dataAnnotationObject.getClassAttributes(), YoloV8ObjectBO.ObjectBO.class);
            return "BOUNDING_BOX".equalsIgnoreCase(objectBO.getType());
        }).collect(Collectors.groupingBy(DataAnnotationObject::getDataId));
        
        modelDatasetResults.forEach(modelDatasetResult -> {
            var isSuccess = modelDatasetResult.getIsSuccess();
            if (!isSuccess) {
                return;
            }
            var modelResult = modelDatasetResult.getModelResult();
            var dataId = modelDatasetResult.getDataId();
            var dataAnnotationObjects = dataAnnotationObjectMap.get(modelDatasetResult.getDataId());
            var groundTruthObjects = new ArrayList<YoloV8Object>();
            if (CollUtil.isEmpty(dataAnnotationObjects)) {
                return;
            }
            dataAnnotationObjects.forEach(dataAnnotationObject -> {
                var yolov8Object = new YoloV8Object();
                var objectBO = DefaultConverter.convert(dataAnnotationObject.getClassAttributes().get("contour"), YoloV8ObjectBO.ObjectBO.class);
                assembleObject(objectBO, yolov8Object);
                groundTruthObjects.add(yolov8Object);
            });
            var modelRunObjects = new ArrayList<YoloV8Object>();
            var yolov8ModelObjectBO = DefaultConverter.convert(modelResult, YoloV8ObjectBO.class);
            yolov8ModelObjectBO.getObjects().forEach(objectBO -> {
                var yolov8Object = new YoloV8Object();
                var confidence = objectBO.getConfidence();
                assembleObject(objectBO, yolov8Object);
                yolov8Object.setConfidence(confidence);
                modelRunObjects.add(yolov8Object);
            });
            var groundTruthObject = YoloV8MetricsReqDTO.builder().id(dataId).objects(groundTruthObjects).build();
            var modelRunObject = YoloV8MetricsReqDTO.builder().id(dataId).objects(modelRunObjects).build();
            FileUtil.appendUtf8String(StrUtil.removeAllLineBreaks(JSONUtil.toJsonStr(groundTruthObject)), groundTruthFilePath);
            FileUtil.appendUtf8String("\n", groundTruthFilePath);
            FileUtil.appendUtf8String(StrUtil.removeAllLineBreaks(JSONUtil.toJsonStr(modelRunObject)), modelRunFilePath);
            FileUtil.appendUtf8String("\n", modelRunFilePath);
        });
    }

    private void assembleObject(YoloV8ObjectBO.ObjectBO objectBO, YoloV8Object yolov8Object) {
        var boundingBox = objectBO.getBoundingBox();
        if (boundingBox == null) {
            return;
        }
        yolov8Object.setLeftTopX(boundingBox.getX());
        yolov8Object.setLeftTopY(boundingBox.getY());
        yolov8Object.setRightBottomX(boundingBox.getX() + boundingBox.getWidth());
        yolov8Object.setRightBottomY(boundingBox.getY() + boundingBox.getHeight());
        yolov8Object.setClassId(objectBO.getClassId());
        yolov8Object.setClassName(objectBO.getClassName());
    }

    @Override
    public String getResultEvaluateUrl() {
        return resultEvaluateUrl;
    }

    @Override
    public ModelCodeEnum getModelCodeEnum() {
        return ModelCodeEnum.YOLOV8_IMAGENET;
    }
} 