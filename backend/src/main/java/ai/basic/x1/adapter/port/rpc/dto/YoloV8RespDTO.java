package ai.basic.x1.adapter.port.rpc.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.util.List;

/**
 * YOLOv8 ImageNet Response DTO
 * @author AI Assistant
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class YoloV8RespDTO {
    private Long id;
    private BigDecimal confidence;
    private List<YoloV8DetectionObject> objects;
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class YoloV8DetectionObject {
        private String classId;
        private String className;
        private BigDecimal confidence;
        private YoloV8BoundingBox boundingBox;
    }
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class YoloV8BoundingBox {
        private BigDecimal x;
        private BigDecimal y;
        private BigDecimal width;
        private BigDecimal height;
    }
} 