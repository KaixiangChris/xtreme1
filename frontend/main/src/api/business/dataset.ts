import { defHttp } from '/@/utils/http/axios';
import {
  ListParams,
  DatasetParams,
  CreateParams,
  InsertUploadDataParams,
  DatasetItem,
  DatasetGetResultModel,
  DatasetListGetResultModel,
  DatasetListItem,
  DatasetIdParams,
  InsertPointCloudParams,
  MakeFrameParams,
  MergeFrameParams,
  GetFrameParams,
  FrameListResult,
  MinioInfo,
  takeRecordParams,
  exportFileRecord,
  GetPresignedParams,
  ResponsePresignedParams,
  UploadParams,
  ResponseUploadRecord,
} from './model/datasetModel';
import { BasicIdParams } from '/@/api/model/baseModel';

enum Api {
  DATASET = '/dataset',
  DATA = '/data',
}

/**
 * @description: Get sample list value
 */

export const datasetListApi = (params: ListParams) =>
  defHttp.get<DatasetListGetResultModel>({
    url: `${Api.DATASET}/findByPage`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const datasetApi = (params: DatasetParams) =>
  defHttp.get<DatasetGetResultModel>({
    url: `${Api.DATA}/findByPage`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const datasetDetailApi = (params: BasicIdParams) =>
  defHttp.get<DatasetItem>({
    url: `${Api.DATA}/info/${params.id}`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const createDatasetApi = (params: CreateParams) =>
  defHttp.post<DatasetItem>({
    url: `${Api.DATASET}/create`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const deleteDatasetApi = (params: BasicIdParams) =>
  defHttp.post<null>({
    url: `${Api.DATASET}/delete/${params.id}`,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const insertUploadData = (params: InsertUploadDataParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/upload`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const insertUploadPointCloudApi = (params: InsertPointCloudParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/uploadCompressed`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const deleteBatchDataset = (params: { ids: number[] }) =>
  defHttp.post<null>({
    url: `${Api.DATA}/deleteBatch`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const updateDataset = (params: { id: string | number; name: string }) =>
  defHttp.post<null>({
    url: `${Api.DATASET}/update/${params.id}`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const datasetItemDetail = (params: BasicIdParams) =>
  defHttp.get<DatasetListItem>({
    url: `${Api.DATASET}/info/${params.id}`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const getMaxCountApi = (params: DatasetIdParams) =>
  defHttp.get<number>({
    url: `${Api.DATA}/selectMaxAnnotationCountByDatasetId`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const makeFrameSeriesApi = (params: MakeFrameParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/frames/combine`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const ungroupFrameSeriesApi = (params: MakeFrameParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/frames/remove`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const mergeFrameApi = (params: MergeFrameParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/frames/merge`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const getFrameApi = (params: GetFrameParams) =>
  defHttp.get<FrameListResult>({
    url: `${Api.DATA}/frames/list`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const getMinioInfo = () =>
  defHttp.get<MinioInfo>({
    url: `${Api.DATASET}/getMinioUserInfo`,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const getLockedByDataset = (params) =>
  defHttp.get<Nullable<any>>({
    url: `${Api.DATA}/findLockRecordIdByDatasetId`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const unLock = (params) =>
  defHttp.post<null>({
    url: `${Api.DATA}/unLock/${params.id}`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const takeRecordByData = (params: takeRecordParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/annotate`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const takeRecordByDataModel = (params: takeRecordParams) =>
  defHttp.post<null>({
    url: `${Api.DATA}/annotateWithModel`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const exportData = (params: any) =>
  defHttp.get<null>({
    url: `${Api.DATA}/export`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const exportDataRecordCallBack = (params: { serialNumbers: string }) =>
  defHttp.get<exportFileRecord[]>({
    url: `${Api.DATA}/findExportRecordBySerialNumbers`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const generatePresignedUrl = (params: GetPresignedParams) =>
  defHttp.get<ResponsePresignedParams>({
    url: `${Api.DATA}/generatePresignedUrl`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const uploadDatasetApi = (params: UploadParams, signal?: any) =>
  defHttp.post<string>({
    url: `${Api.DATA}/upload`,
    signal: signal,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const findUploadRecordBySerialNumbers = (params: string, signal?: any) =>
  defHttp.get<ResponseUploadRecord[]>({
    url: `${Api.DATA}/findUploadRecordBySerialNumbers?serialNumbers=${params}`,
    signal: signal,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const getStatusNum = (params: { datasetId: number }) =>
  defHttp.get<ResponseUploadRecord[]>({
    url: `${Api.DATA}/getAnnotationStatusStatisticsByDatasetId`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });

export const hasOntologyApi = (params: { datasetId: number }) =>
  defHttp.get<ResponseUploadRecord[]>({
    url: `${Api.DATASET}/findOntologyIsExistByDatasetId`,
    params,
    headers: {
      // @ts-ignore
      ignoreCancelToken: true,
    },
  });
