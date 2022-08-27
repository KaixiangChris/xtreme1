package ai.basic.x1.adapter.dto;

import ai.basic.x1.entity.enums.ToolTypeEnum;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import javax.validation.groups.Default;

/**
 * @author chenchao
 * @date 2022/8/24
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DatasetClassDTO {

    private Long id;

    /**
     * The id of the inherited ontology
     */
    private Long ontologyId;

    /**
     * The id of the inherited classification in the ontology
     */
    private Long classId;

    @NotNull(message = "not allowed null!",groups = GroupSave.class)
    private Long datasetId;

    @NotEmpty(message = "not allowed null!",groups = GroupSave.class)
    @Length(max = 256,message = "The length of name should be less than 256.")
    private String name;

    @NotEmpty(message = "not allowed null!",groups = GroupSave.class)
    private String color;

    private ToolTypeEnum toolType;

    private JSONObject toolTypeOptions;

    private JSONArray attributes;

    private Integer datasetClassNum;

    /**
     * dedicated business logic groupings
     */
    public interface GroupSave extends Default {
    }
}
