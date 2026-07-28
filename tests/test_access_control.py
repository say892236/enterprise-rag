# tests/test_access_control.py

"""
权限控制单元测试。

测试目标：
1. 不依赖 FastAPI 服务。
2. 不调用 RAG、Embedding 或 LLM。
3. 只测试 api/access_control.py 中的权限判断逻辑。
4. 验证公共文档、部门私有文档、管理员权限和越权拦截。
5. 自动读取当前权限配置，避免测试文件与文档名称再次不同步。
"""

import sys
import traceback
from collections.abc import Callable

from api.access_control import (
    DEPARTMENT_SOURCE_MAP,
    PUBLIC_SOURCES,
    PermissionDeniedError,
    build_effective_metadata_filter,
    get_allowed_sources,
)


def print_pass(message: str) -> None:
    """打印测试通过信息。"""

    print(f"[PASS] {message}")


def assert_permission_denied(
    func: Callable[[], object],
    message: str,
) -> None:
    """
    断言指定操作会抛出 PermissionDeniedError。
    """

    try:
        func()
    except PermissionDeniedError:
        print_pass(message)
        return

    raise AssertionError(
        f"预期抛出 PermissionDeniedError，但实际没有抛出：{message}"
    )


def get_private_source(role: str) -> str:
    """
    获取指定部门的一份私有文档，用于权限测试。

    测试直接读取当前权限配置，不再写死旧文件名。
    """

    private_sources = DEPARTMENT_SOURCE_MAP.get(role)

    assert private_sources, (
        f"角色 {role} 没有配置部门私有文档，"
        "无法执行权限测试"
    )

    return private_sources[0]


def test_employee_allowed_sources() -> None:
    """
    测试普通员工只能访问公共文档。
    """

    allowed_sources = get_allowed_sources("employee")

    assert allowed_sources is not None

    # employee 的权限应与公共文档列表完全一致
    assert allowed_sources == list(PUBLIC_SOURCES)

    # 普通员工不能访问任何部门私有文档
    for role, private_sources in DEPARTMENT_SOURCE_MAP.items():
        for source in private_sources:
            assert source not in allowed_sources, (
                f"employee 不应访问 {role} 私有文档：{source}"
            )

    print_pass("employee 只能访问公共文档")


def test_department_allowed_sources() -> None:
    """
    测试所有部门角色均可访问：

    公共文档 + 本部门私有文档

    同时不能访问其他部门的私有文档。
    """

    for role, own_private_sources in DEPARTMENT_SOURCE_MAP.items():
        allowed_sources = get_allowed_sources(role)

        assert allowed_sources is not None

        # 当前角色必须拥有全部公共文档权限
        for public_source in PUBLIC_SOURCES:
            assert public_source in allowed_sources, (
                f"{role} 缺少公共文档权限：{public_source}"
            )

        # 当前角色必须拥有本部门全部私有文档权限
        for private_source in own_private_sources:
            assert private_source in allowed_sources, (
                f"{role} 缺少本部门文档权限：{private_source}"
            )

        # 当前角色不能访问其他部门的私有文档
        for other_role, other_private_sources in DEPARTMENT_SOURCE_MAP.items():
            if other_role == role:
                continue

            for source in other_private_sources:
                assert source not in allowed_sources, (
                    f"{role} 不应访问 {other_role} 私有文档：{source}"
                )

        # 权限列表中不应该出现重复文档
        assert len(allowed_sources) == len(set(allowed_sources)), (
            f"{role} 的文档权限列表存在重复项"
        )

    print_pass("所有部门角色只能访问公共文档和本部门私有文档")


def test_admin_allowed_sources() -> None:
    """
    测试 admin 不限制文档访问范围。

    返回 None 表示不追加 source 白名单限制。
    """

    allowed_sources = get_allowed_sources("admin")

    assert allowed_sources is None

    print_pass("admin 不限制文档访问范围")


def test_employee_default_metadata_filter() -> None:
    """
    测试普通员工未传 metadata_filter 时，
    系统自动限制为公共文档。
    """

    metadata_filter = build_effective_metadata_filter(
        user_role="employee",
        request_metadata_filter=None,
    )

    assert metadata_filter is not None
    assert metadata_filter == {
        "source": {
            "$in": list(PUBLIC_SOURCES),
        }
    }

    print_pass("employee 未传 metadata_filter 时自动限制为公共文档")


def test_department_default_metadata_filter() -> None:
    """
    测试部门角色未指定 source 时，
    系统自动加入公共文档和本部门私有文档。
    """

    for role, private_sources in DEPARTMENT_SOURCE_MAP.items():
        metadata_filter = build_effective_metadata_filter(
            user_role=role,
            request_metadata_filter=None,
        )

        assert metadata_filter is not None

        actual_sources = metadata_filter["source"]["$in"]

        expected_sources = list(
            dict.fromkeys(
                [
                    *PUBLIC_SOURCES,
                    *private_sources,
                ]
            )
        )

        assert actual_sources == expected_sources, (
            f"{role} 自动生成的 source 权限范围不正确"
        )

    print_pass("部门角色自动限制为公共文档和本部门私有文档")


def test_department_can_access_own_private_doc() -> None:
    """
    测试所有部门角色都可以精确访问本部门私有文档。
    """

    for role in DEPARTMENT_SOURCE_MAP:
        private_source = get_private_source(role)

        metadata_filter = build_effective_metadata_filter(
            user_role=role,
            request_metadata_filter={
                "source": private_source,
            },
        )

        assert metadata_filter == {
            "source": private_source,
        }

    print_pass("所有部门角色可以访问本部门私有文档")


def test_employee_cannot_access_private_doc() -> None:
    """
    测试普通员工不能访问任何部门私有文档。
    """

    for role in DEPARTMENT_SOURCE_MAP:
        private_source = get_private_source(role)

        assert_permission_denied(
            lambda source=private_source: build_effective_metadata_filter(
                user_role="employee",
                request_metadata_filter={
                    "source": source,
                },
            ),
            f"employee 不能访问 {role} 私有文档",
        )


def test_cross_department_access_denied() -> None:
    """
    测试部门之间不能互相访问私有文档。
    """

    roles = list(DEPARTMENT_SOURCE_MAP.keys())

    assert len(roles) >= 2, "至少需要配置两个部门角色才能测试跨部门权限"

    for index, current_role in enumerate(roles):
        # 选择下一个部门作为越权目标
        target_role = roles[(index + 1) % len(roles)]
        target_source = get_private_source(target_role)

        assert_permission_denied(
            lambda role=current_role, source=target_source: (
                build_effective_metadata_filter(
                    user_role=role,
                    request_metadata_filter={
                        "source": source,
                    },
                )
            ),
            f"{current_role} 不能访问 {target_role} 私有文档",
        )


def test_source_in_filter_allowed_sources() -> None:
    """
    测试 source.$in 全部属于当前角色权限时可以正常使用。
    """

    hr_private_source = get_private_source("hr")
    public_source = PUBLIC_SOURCES[0]

    requested_sources = [
        public_source,
        hr_private_source,
    ]

    metadata_filter = build_effective_metadata_filter(
        user_role="hr",
        request_metadata_filter={
            "source": {
                "$in": requested_sources,
            }
        },
    )

    assert metadata_filter == {
        "source": {
            "$in": requested_sources,
        }
    }

    print_pass("source.$in 全部有权限时可以正常查询")


def test_source_in_filter_mixed_permissions_denied() -> None:
    """
    测试 source.$in 同时包含有权限和无权限文档时，
    整个请求应被拒绝。

    当前权限策略采用严格模式：
    不会静默删除越权文档。
    """

    public_source = PUBLIC_SOURCES[0]
    hr_private_source = get_private_source("hr")
    finance_private_source = get_private_source("finance")

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="hr",
            request_metadata_filter={
                "source": {
                    "$in": [
                        public_source,
                        hr_private_source,
                        finance_private_source,
                    ]
                }
            },
        ),
        "source.$in 包含任意越权文档时拒绝整个请求",
    )


def test_source_in_filter_all_denied() -> None:
    """
    测试 source.$in 中全部都是无权限文档时直接拒绝。
    """

    hr_private_source = get_private_source("hr")
    finance_private_source = get_private_source("finance")

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="employee",
            request_metadata_filter={
                "source": {
                    "$in": [
                        hr_private_source,
                        finance_private_source,
                    ]
                }
            },
        ),
        "source.$in 全部无权限时返回 PERMISSION_DENIED",
    )


def test_source_in_filter_must_be_list() -> None:
    """
    测试 source.$in 必须是列表。
    """

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="employee",
            request_metadata_filter={
                "source": {
                    "$in": "差旅住宿与交通标准.md",
                }
            },
        ),
        "source.$in 不是列表时拒绝请求",
    )


def test_source_in_filter_cannot_be_empty() -> None:
    """
    测试 source.$in 不能为空列表。
    """

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="employee",
            request_metadata_filter={
                "source": {
                    "$in": [],
                }
            },
        ),
        "source.$in 为空列表时拒绝请求",
    )


def test_unsupported_source_filter_denied() -> None:
    """
    测试不支持的 source 过滤表达式会被拒绝，
    防止通过其他操作符绕过权限控制。
    """

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="employee",
            request_metadata_filter={
                "source": {
                    "$ne": "信息安全与数据保密制度.md",
                }
            },
        ),
        "不支持的 source 过滤格式会被拒绝",
    )


def test_other_metadata_filter_is_preserved() -> None:
    """
    测试系统追加 source 权限时，
    不会删除请求中的其他 metadata 条件。
    """

    metadata_filter = build_effective_metadata_filter(
        user_role="employee",
        request_metadata_filter={
            "file_type": "markdown",
        },
    )

    assert metadata_filter is not None
    assert metadata_filter["file_type"] == "markdown"
    assert metadata_filter["source"]["$in"] == list(PUBLIC_SOURCES)

    print_pass("追加 source 权限时保留其他 metadata 条件")


def test_admin_keep_request_filter() -> None:
    """
    测试 admin 保留用户传入的 metadata_filter，
    不额外追加 source 白名单。
    """

    request_filter = {
        "source": get_private_source("hr"),
        "file_type": "markdown",
    }

    metadata_filter = build_effective_metadata_filter(
        user_role="admin",
        request_metadata_filter=request_filter,
    )

    assert metadata_filter == request_filter

    print_pass("admin 保留原始 metadata_filter")


def test_admin_without_filter_returns_none() -> None:
    """
    测试 admin 未传过滤条件时返回 None。
    """

    metadata_filter = build_effective_metadata_filter(
        user_role="admin",
        request_metadata_filter=None,
    )

    assert metadata_filter is None

    print_pass("admin 未传 metadata_filter 时不限制文档来源")


def test_unknown_role_denied() -> None:
    """
    测试未知角色会被拒绝。
    """

    assert_permission_denied(
        lambda: get_allowed_sources("unknown_role"),
        "未知角色会被拒绝",
    )


def run_all_tests() -> None:
    """
    按顺序运行权限控制测试。
    """

    print("=" * 100)
    print("开始运行权限控制单元测试")
    print("=" * 100)

    test_employee_allowed_sources()
    test_department_allowed_sources()
    test_admin_allowed_sources()

    test_employee_default_metadata_filter()
    test_department_default_metadata_filter()

    test_department_can_access_own_private_doc()
    test_employee_cannot_access_private_doc()
    test_cross_department_access_denied()

    test_source_in_filter_allowed_sources()
    test_source_in_filter_mixed_permissions_denied()
    test_source_in_filter_all_denied()
    test_source_in_filter_must_be_list()
    test_source_in_filter_cannot_be_empty()
    test_unsupported_source_filter_denied()

    test_other_metadata_filter_is_preserved()

    test_admin_keep_request_filter()
    test_admin_without_filter_returns_none()
    test_unknown_role_denied()

    print("=" * 100)
    print("权限控制单元测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as exc:
        print(
            f"[FAIL] {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        sys.exit(1)