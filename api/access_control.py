"""企业知识库的文档来源权限控制。"""



class PermissionDeniedError(Exception):
    """
    用户访问无权限文档时抛出。
    """

    pass




# 全体员工都可以访问的公共制度文档
PUBLIC_SOURCES = [
    "company_kb.md",
    "leave_policy.md",
    "employee_handbook.md",
    "travel_policy.md",
    "it_security_policy.md",
    "reimbursement_policy.md",
]


# 部门专属文档
# 当前你还没有真正的部门私有文档，可以先留空或后面补
DEPARTMENT_SOURCE_MAP = {
    "hr": [

        "hr_salary_policy.md",

    ],

    "finance": [

        "finance_budget_policy.md",

    ],

    "it": [

        "it_admin_policy.md",

    ],
}

def get_allowed_sources(user_role: str) -> list[str] | None:
    """返回角色允许访问的 source；``None`` 表示不限制来源。"""

    role = user_role or "employee"
    if role == "admin":
        return None
    if role == "employee":
        return list(PUBLIC_SOURCES)
    if role not in DEPARTMENT_SOURCE_MAP:
        raise PermissionDeniedError(f"未知用户角色: {role}")

    # 返回新列表，避免调用方意外修改权限配置。
    return [*PUBLIC_SOURCES, *DEPARTMENT_SOURCE_MAP[role]]


def build_effective_metadata_filter(
    user_role: str,
    request_metadata_filter: dict | None = None,
) -> dict | None:
    """构建在请求过滤条件上叠加 source 权限后的最终过滤条件。

    ``source`` 支持精确字符串和 ``{"$in": [...]}`` 两种形式。管理员的
    请求不受 source 权限限制；其他角色只能查询其允许的文档来源。
    """

    request_filter = request_metadata_filter or {}
    allowed_sources = get_allowed_sources(user_role)

    if allowed_sources is None:  # admin
        return request_filter or None

    if "source" not in request_filter:
        return {**request_filter, "source": {"$in": allowed_sources}}

    requested_source = request_filter["source"]
    if isinstance(requested_source, str):
        if requested_source not in allowed_sources:
            raise PermissionDeniedError(f"当前角色无权访问文档: {requested_source}")
        return dict(request_filter)

    if isinstance(requested_source, dict) and set(requested_source) == {"$in"}:
        requested_sources = requested_source["$in"]
        if not isinstance(requested_sources, list):
            raise PermissionDeniedError("source.$in 必须是列表")

        permitted_sources = [
            source for source in requested_sources if source in allowed_sources
        ]
        if not permitted_sources:
            raise PermissionDeniedError("当前角色无权访问请求中的任何文档")

        return {**request_filter, "source": {"$in": permitted_sources}}

    raise PermissionDeniedError(f"不支持的 source 过滤格式: {requested_source}")

