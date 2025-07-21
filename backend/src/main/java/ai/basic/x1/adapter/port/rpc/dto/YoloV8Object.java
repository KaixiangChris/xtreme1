package ai.basic.x1.adapter.port.rpc.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;

/**
 * YOLOv8 Object DTO for metrics calculation
 * @author AI Assistant
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class YoloV8Object {
    private String classId;
    private String className;
    private BigDecimal confidence;
    private BigDecimal leftTopX;
    private BigDecimal leftTopY;
    private BigDecimal rightBottomX;
    private BigDecimal rightBottomY;
} 