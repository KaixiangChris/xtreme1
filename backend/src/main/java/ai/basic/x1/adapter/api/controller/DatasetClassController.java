package ai.basic.x1.adapter.api.controller;

import ai.basic.x1.adapter.dto.DatasetClassDTO;
import ai.basic.x1.entity.DatasetClassBO;
import ai.basic.x1.usecase.DatasetClassUseCase;
import ai.basic.x1.usecase.exception.UsecaseCode;
import ai.basic.x1.usecase.exception.UsecaseException;
import ai.basic.x1.util.DefaultConverter;
import ai.basic.x1.util.Page;
import ai.basic.x1.util.lock.IDistributedLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import javax.validation.groups.Default;
import java.util.List;

/**
 * @author chenchao
 * @date 2022/3/11
 */
@RestController
@RequestMapping("/datasetClass/")
public class DatasetClassController {

    private final DatasetClassUseCase datasetClassUseCase;

    @Autowired
    private IDistributedLock distributedLock;

    public DatasetClassController(DatasetClassUseCase datasetClassUseCase) {
        this.datasetClassUseCase = datasetClassUseCase;
    }

    @PostMapping("create")
    public void create(@Validated({Default.class, DatasetClassDTO.GroupSave.class}) @RequestBody DatasetClassDTO datasetClassDTO) {
        save(datasetClassDTO);
    }

    @PostMapping("update/{id}")
    public void update(@PathVariable Long id, @Validated({Default.class, DatasetClassDTO.GroupSave.class}) @RequestBody DatasetClassDTO datasetClassDTO) {
        datasetClassDTO.setId(id);
        save(datasetClassDTO);
    }

    @PostMapping("delete/{id}")
    public void delete(@PathVariable("id") Long id) {
        datasetClassUseCase.deleteClass(id);
    }

    @GetMapping("info/{id}")
    public DatasetClassDTO info(@PathVariable("id") Long id) {
        return DefaultConverter.convert(datasetClassUseCase.findById(id), DatasetClassDTO.class);
    }

    @GetMapping("findByPage")
    public Page<DatasetClassDTO> findByPage(@RequestParam(defaultValue = "1") Integer pageNo,
                                            @RequestParam(defaultValue = "10") Integer pageSize,
                                            DatasetClassDTO datasetClassReqDTO) {
        DatasetClassBO datasetClassBO = DefaultConverter.convert(datasetClassReqDTO, DatasetClassBO.class);
        Assert.notNull(datasetClassBO.getDatasetId(), "datasetId can not be null");
        return DefaultConverter.convert(datasetClassUseCase.findByPage(pageNo,
                pageSize, datasetClassBO), DatasetClassDTO.class);
    }

    @GetMapping("findAll/{datasetId}")
    public List<DatasetClassDTO> findAll(@PathVariable Long datasetId) {
        return DefaultConverter.convert(datasetClassUseCase.findAll(datasetId), DatasetClassDTO.class);
    }

    /**
     * Check whether the class name already exists in the same dataset
     * @return if exists return true
     */
    @GetMapping("validateName")
    public Boolean validateName(@RequestParam Long datasetId, @RequestParam String name,@RequestParam(required = false) Long id) {
        return datasetClassUseCase.nameExists(DatasetClassBO.builder().datasetId(datasetId).name(name).id(id).build());
    }

    private void save(DatasetClassDTO dto) {
        DatasetClassBO datasetClassBO = DefaultConverter.convert(dto, DatasetClassBO.class);
        var lockKey = String.format("%s:%s:%s", "datasetClass", "datasetId+className", datasetClassBO.getDatasetId() + "+" + datasetClassBO.getName());
        var boo = distributedLock.tryLock(lockKey, 1000);
        try {
            if (!boo) {
                throw new UsecaseException(UsecaseCode.NAME_DUPLICATED);
            }
            datasetClassUseCase.saveDatasetClass(datasetClassBO);
        } catch (Exception e) {
            throw e;
        } finally {
            distributedLock.unlock(lockKey);
        }
    }
}
