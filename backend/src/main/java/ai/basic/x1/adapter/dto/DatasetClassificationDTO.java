package ai.basic.x1.adapter.dto;

import ai.basic.x1.entity.enums.InputTypeEnum;
import ai.basic.x1.entity.enums.SortByEnum;
import ai.basic.x1.entity.enums.SortEnum;
import cn.hutool.json.JSONArray;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotNull;

/**
 * @author chenchao
 * @date 2022/4/11
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
public class DatasetClassificationDTO {

    private Long id;

    /**
     * The id of the inherited ontology
     */
    private Long ontologyId;

    /**
     * The id of the inherited classification in the ontology
     */
    private Long classificationId;

    @NotNull(groups = GroupSave.class)
    private Long datasetId;

    @NotNull(groups = GroupSave.class)
    @Length(max = 256, message = "The length of name should be less than 256.")
    private String name;

    private JSONArray options;

    /**
     * input type:'RADIO','MULTI_SELECTION','DROPDOWN','TEXT'
     */
    private InputTypeEnum inputType;

    private Boolean isRequired;


    /**
     * Query value, create start time
     */
    private String startTime;

    /**
     * Query value, create end time
     */
    private String endTime;

    private SortByEnum sortBy;

    private SortEnum ascOrDesc;


    public interface GroupSave {

    }
}
