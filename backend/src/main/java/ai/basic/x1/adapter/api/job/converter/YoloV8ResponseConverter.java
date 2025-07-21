package ai.basic.x1.adapter.api.job.converter;

import ai.basic.x1.adapter.dto.ApiResult;
import ai.basic.x1.adapter.dto.PreModelParamDTO;
import ai.basic.x1.adapter.port.dao.mybatis.model.ModelClass;
import ai.basic.x1.adapter.port.rpc.dto.YoloV8RespDTO;
import ai.basic.x1.entity.YoloV8ObjectBO;
import ai.basic.x1.usecase.exception.UsecaseCode;
import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONUtil;
import lombok.extern.slf4j.Slf4j;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * YOLOv8 Response Converter
 * @author AI Assistant
 */
@Slf4j
public class YoloV8ResponseConverter {

    public static YoloV8ObjectBO convert(ApiResult<YoloV8RespDTO> yolov8RespDTOApiResult,
                                        Map<String, ModelClass> systemModelClassMap,
                                        PreModelParamDTO filterCondition) {
        YoloV8ObjectBO.YoloV8ObjectBOBuilder<?, ?> builder = YoloV8ObjectBO.builder();
        var response = yolov8RespDTOApiResult.getData();
        if (yolov8RespDTOApiResult.getCode() == UsecaseCode.OK) {
            if (CollUtil.isEmpty(response.getObjects())) {
                builder.code(UsecaseCode.OK.getCode())
                        .message("success")
                        .dataId(response.getId())
                        .objects(Collections.emptyList());
            } else {
                log.info("start filter YOLOv8 predItem. filter condition: " + JSONUtil.toJsonStr(filterCondition));
                var predObjects = response.getObjects()
                        .stream()
                        .filter(item -> matchSelectedClassAndConfidence(item, filterCondition))
                        .map(item -> buildObject(item, systemModelClassMap))
                        .collect(Collectors.toList());
                builder.confidence(response.getConfidence());
                if (CollUtil.isNotEmpty(predObjects) && ObjectUtil.isNull(response.getConfidence())) {
                    var dataConfidence = predObjects.stream().mapToDouble(object -> object.getConfidence().doubleValue()).summaryStatistics();
                    builder.confidence(BigDecimal.valueOf(dataConfidence.getAverage()));
                }
                builder.code(UsecaseCode.OK.getCode())
                        .message("success")
                        .dataId(response.getId())
                        .objects(predObjects);
            }
        } else {
            builder.code(yolov8RespDTOApiResult.getCode().toString())
                    .message(yolov8RespDTOApiResult.getMessage())
                    .dataId(response != null ? response.getId() : null)
                    .objects(Collections.emptyList());
        }
        return builder.build();
    }

    private static boolean matchSelectedClassAndConfidence(YoloV8RespDTO.YoloV8DetectionObject item, PreModelParamDTO filterCondition) {
        if (filterCondition == null) {
            return true;
        }
        
        // Check confidence threshold
        if (filterCondition.getMinConfidence() != null && 
            item.getConfidence().compareTo(filterCondition.getMinConfidence()) < 0) {
            return false;
        }
        
        if (filterCondition.getMaxConfidence() != null && 
            item.getConfidence().compareTo(filterCondition.getMaxConfidence()) > 0) {
            return false;
        }
        
        // Check class filter
        if (CollUtil.isNotEmpty(filterCondition.getClasses())) {
            return filterCondition.getClasses().contains(item.getClassId());
        }
        
        return true;
    }

    private static YoloV8ObjectBO.ObjectBO buildObject(YoloV8RespDTO.YoloV8DetectionObject item, Map<String, ModelClass> systemModelClassMap) {
        var boundingBox = YoloV8ObjectBO.BoundingBoxBO.builder()
                .x(item.getBoundingBox().getX())
                .y(item.getBoundingBox().getY())
                .width(item.getBoundingBox().getWidth())
                .height(item.getBoundingBox().getHeight())
                .build();

        // Convert bounding box to points for compatibility
        var points = List.of(
                YoloV8ObjectBO.PointBO.builder()
                        .x(item.getBoundingBox().getX())
                        .y(item.getBoundingBox().getY())
                        .build(),
                YoloV8ObjectBO.PointBO.builder()
                        .x(item.getBoundingBox().getX().add(item.getBoundingBox().getWidth()))
                        .y(item.getBoundingBox().getY().add(item.getBoundingBox().getHeight()))
                        .build()
        );

        return YoloV8ObjectBO.ObjectBO.builder()
                .classId(item.getClassId())
                .className(item.getClassName())
                .confidence(item.getConfidence())
                .type("BOUNDING_BOX")
                .boundingBox(boundingBox)
                .points(points)
                .build();
    }
} 