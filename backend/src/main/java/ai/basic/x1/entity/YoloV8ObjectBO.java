package ai.basic.x1.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.math.BigDecimal;
import java.util.List;

/**
 * YOLOv8 ImageNet Object Business Object
 * @author AI Assistant
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
public class YoloV8ObjectBO extends ModelTaskInfoBO {
    private Long dataId;
    private List<ObjectBO> objects;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ObjectBO {
        private String classId;
        private String className;
        private BigDecimal confidence;
        private String type;
        private BoundingBoxBO boundingBox;
        private List<PointBO> points;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BoundingBoxBO {
        private BigDecimal x;
        private BigDecimal y;
        private BigDecimal width;
        private BigDecimal height;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PointBO {
        private BigDecimal x;
        private BigDecimal y;
    }
} 