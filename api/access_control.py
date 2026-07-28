"""企业知识库的文档来源权限控制。"""



class PermissionDeniedError(Exception):
    """
    用户访问无权限文档时抛出。
    """

    pass




# 全体员工都可以访问的公共制度文档
PUBLIC_SOURCES = [
    "公司概况与组织信息.md",
    "企业文化与行为规范.md",
    "会议室预约与使用规范.md",
    "员工手册.md",
    "差旅住宿与交通标准.md",
    "信息安全与数据保密制度.md",
    "办公物资申领管理规定.md",
    "办公设备领用与归还规范.md",
    "合同发起与签署流程.md"
    "员工奖惩管理办法.md",
    "绩效考核制度(员工版).md",
    "考勤与休假管理制度.md",
    "费用报销制度与流程.md",
    "采购申请与审批流程.md",
]


# 部门专属文档
#
# 说明：
# 1. key 使用系统中的用户角色名称。
# 2. value 是该角色额外允许访问的部门私有文档。
# 3. PUBLIC_SOURCES 会在权限构建时统一追加，不需要重复写进这里。
# 4. admin 超级管理员应在权限逻辑中单独放行全部文档。
DEPARTMENT_SOURCE_MAP = {
    # 人力资源部门
    "hr": [
        "员工入职办理流程(HR内部版).md",
        "员工档案管理规范.md",
        "招聘面试流程与评估细则.md",
        "社保公积金内部办理指引.md",
        "离职面谈与交接HR执行规范.md",
        "绩效结果校准与强制分布细则.md",
        "薪酬体系与调薪管理制度.md",
    ],

    # 财务部门
    "finance": [
        "付款审批权限与异常处理.md",
        "发票开具与税务申报操作手册.md",
        "工资核算与发放流程.md",
        "费用报销财务审核标准.md",
        "预算执行监控与调整流程.md",
    ],

    # 研发部门
    "engineering": [
        "API接口设计规范.md",
        "Git分支策略与提交规范.md",
        "上线发布Checklist与回滚方案.md",
        "事故分级与复盘改进流程.md",
        "代码规范与Code Review流程.md",
        "技术选型与架构评审指南.md",
        "数据备份与容灾恢复预案.md",
        "测试管理与质量门禁规范.md",
        "研发环境配置规范.md",
        "账号权限开通与审批流程（内部版）.md",
    ],

    # 行政部门
    #
    # 注意：
    # administration 表示行政部门角色，
    # 不要与超级管理员角色 admin 混淆。
    "administration": [
        "办公环境与保洁管理规范.md",
        "印章保管与用印登记规范.md",
        "员工入职行政办理流程.md",
        "固定资产全生命周期管理.md",
        "访客接待管理规范.md",
        "车辆调度与司机管理细则.md",
    ],

    # 法务部门
    "legal": [
        "保密协议管理规范.md",
        "合同审查标准条款清单.md",
        "合同风险审核流程.md",
        "商业合规审计与风险控制细则.md",
        "知识产权注册与维护指引.md",
        "诉讼与仲裁案件管理流程.md",
    ],

    # 管理层
    "management": [
        "人员晋升审批流程.md",
        "员工离职审批流程.md",
        "员工绩效辅导流程.md",
        "团队管理规范.md",
        "管理者权限与职责说明.md",
        "管理者绩效评价指南.md",
    ],
}

def get_allowed_sources(user_role: str) -> list[str] | None:
    """
    返回当前角色允许访问的文档 source。

    返回值：
    - None：不限制文档来源，当前仅 admin 使用。
    - list[str]：当前角色允许访问的文档列表。
    """

    # 清理角色名，避免大小写和首尾空格导致匹配失败
    role = (user_role or "employee").strip().lower()

    # 超级管理员可以访问全部文档
    if role == "admin":
        return None

    # 普通员工只能访问公共文档
    if role == "employee":
        return list(PUBLIC_SOURCES)

    # 未配置的角色直接拒绝，避免意外放行
    if role not in DEPARTMENT_SOURCE_MAP:
        raise PermissionDeniedError(
            f"未知用户角色: {role}"
        )

    # 部门用户可以访问：
    # 公共文档 + 当前部门专属文档
    #
    # dict.fromkeys 用于保持原顺序并去重
    allowed_sources = list(
        dict.fromkeys(
            [
                *PUBLIC_SOURCES,
                *DEPARTMENT_SOURCE_MAP[role],
            ]
        )
    )

    return allowed_sources


def build_effective_metadata_filter(
    user_role: str,
    request_metadata_filter: dict | None = None,
) -> dict | None:
    """
    在用户主动传入的 metadata_filter 上，
    追加当前角色允许访问的 source 权限条件。

    source 支持两种格式：

    1. 精确匹配：
       {"source": "员工手册.md"}

    2. 多文档匹配：
       {"source": {"$in": ["员工手册.md", "差旅制度.md"]}}

    权限规则：
    - admin：不限制 source。
    - 其他角色：只能访问允许列表中的文档。
    """

    # 复制请求过滤条件，避免修改原始请求对象
    request_filter = dict(
        request_metadata_filter or {}
    )

    allowed_sources = get_allowed_sources(
        user_role=user_role,
    )

    # admin 不追加 source 权限限制
    if allowed_sources is None:
        return request_filter or None

    # 用户没有主动指定 source 时，
    # 自动限制为当前角色所有可访问文档
    if "source" not in request_filter:
        return {
            **request_filter,
            "source": {
                "$in": allowed_sources,
            },
        }

    requested_source = request_filter["source"]

    # 情况一：用户指定单个文档
    if isinstance(requested_source, str):
        if requested_source not in allowed_sources:
            raise PermissionDeniedError(
                f"当前角色无权访问文档: {requested_source}"
            )

        return request_filter

    # 情况二：用户指定多个文档
    if (
        isinstance(requested_source, dict)
        and set(requested_source.keys()) == {"$in"}
    ):
        requested_sources = requested_source["$in"]

        if not isinstance(requested_sources, list):
            raise PermissionDeniedError(
                "source.$in 必须是列表"
            )

        if not requested_sources:
            raise PermissionDeniedError(
                "source.$in 不能为空列表"
            )

        # 找出请求中所有没有访问权限的文档
        denied_sources = [
            source
            for source in requested_sources
            if source not in allowed_sources
        ]

        # 只要请求中存在越权文档，就直接拒绝整个请求
        # 避免系统静默删除条件，导致实际查询与用户请求不一致
        if denied_sources:
            raise PermissionDeniedError(
                "当前角色无权访问以下文档: "
                + "、".join(denied_sources)
            )

        # 保留原顺序并去重
        permitted_sources = list(
            dict.fromkeys(requested_sources)
        )

        return {
            **request_filter,
            "source": {
                "$in": permitted_sources,
            },
        }

    # 其他 source 表达式暂不允许，避免绕过权限控制
    raise PermissionDeniedError(
        f"不支持的 source 过滤格式: {requested_source}"
    )
