package ai.basic.x1.adapter.port.rpc;

import ai.basic.x1.adapter.dto.ApiResult;
import ai.basic.x1.adapter.port.rpc.dto.ImageDetectionReqDTO;
import ai.basic.x1.adapter.port.rpc.dto.YoloV8RespDTO;
import ai.basic.x1.usecase.exception.UsecaseCode;
import ai.basic.x1.usecase.exception.UsecaseException;
import cn.hutool.http.ContentType;
import cn.hutool.http.HttpStatus;
import cn.hutool.http.HttpUtil;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

/**
 * YOLOv8 Model HTTP Caller
 * @author AI Assistant
 */
@Component
public class YoloV8ModelHttpCaller {

    @Autowired
    private ObjectMapper objectMapper;

    public ApiResult<List<YoloV8RespDTO>> callYoloV8Model(ImageDetectionReqDTO requestBody, String url) throws IOException {
        var requestBodyStr = objectMapper.writeValueAsString(requestBody);
        var httpRequest = HttpUtil.createPost(url)
                .body(requestBodyStr, ContentType.JSON.getValue());
        var httpResponse = httpRequest.execute();
        ApiResult<List<YoloV8RespDTO>> result;
        if (httpResponse.getStatus() == HttpStatus.HTTP_OK) {
            result = objectMapper.readValue(httpResponse.bodyBytes(),
                    new TypeReference<ApiResult<List<YoloV8RespDTO>>>() {
            });
        } else {
            throw new UsecaseException(UsecaseCode.UNKNOWN, httpResponse.body());
        }
        return result;
    }
} 