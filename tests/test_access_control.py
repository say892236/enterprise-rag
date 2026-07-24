# tests/test_access_control.py

"""
权限控制单元测试。

测试目标：
1. 不依赖 FastAPI 服务。
2. 不调用 RAG / LLM。
3. 只测试 api/access_control.py 里的权限判断逻辑。
4. 验证 employee / hr / finance / it / admin 的文档访问范围。
"""

import sys

from api.access_control import (
    PermissionDeniedError,
    build_effective_metadata_filter,
    get_allowed_sources,
)


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def assert_permission_denied(func, message: str) -> None:
    """
    断言某个操作会抛出 PermissionDeniedError。
    """

    try:
        func()
    except PermissionDeniedError:
        print_pass(message)
        return

    raise AssertionError(f"预期抛出 PermissionDeniedError，但实际没有抛出：{message}")


def test_employee_allowed_sources() -> None:
    """
    测试普通员工只能访问公共文档。
    """

    allowed_sources = get_allowed_sources("employee")

    assert allowed_sources is not None
    assert "employee_handbook.md" in allowed_sources
    assert "leave_policy.md" in allowed_sources
    assert "travel_policy.md" in allowed_sources
    assert "reimbursement_policy.md" in allowed_sources

    # 普通员工不能访问部门私有文档
    assert "hr_salary_policy.md" not in allowed_sources
    assert "finance_budget_policy.md" not in allowed_sources
    assert "it_admin_policy.md" not in allowed_sources

    print_pass("employee 只能访问公共文档")


def test_department_allowed_sources() -> None:
    """
    测试部门用户可以访问公共文档 + 本部门私有文档。
    """

    hr_sources = get_allowed_sources("hr")
    finance_sources = get_allowed_sources("finance")
    it_sources = get_allowed_sources("it")

    assert hr_sources is not None
    assert finance_sources is not None
    assert it_sources is not None

    # HR 可以访问 HR 私有文档
    assert "hr_salary_policy.md" in hr_sources
    assert "finance_budget_policy.md" not in hr_sources
    assert "it_admin_policy.md" not in hr_sources

    # 财务可以访问财务私有文档
    assert "finance_budget_policy.md" in finance_sources
    assert "hr_salary_policy.md" not in finance_sources
    assert "it_admin_policy.md" not in finance_sources

    # IT 可以访问 IT 私有文档
    assert "it_admin_policy.md" in it_sources
    assert "hr_salary_policy.md" not in it_sources
    assert "finance_budget_policy.md" not in it_sources

    print_pass("部门用户只能访问本部门私有文档")


def test_admin_allowed_sources() -> None:
    """
    测试 admin 不限制文档访问范围。

    get_allowed_sources("admin") 返回 None，
    表示不做 source 限制。
    """

    allowed_sources = get_allowed_sources("admin")

    assert allowed_sources is None

    print_pass("admin 不限制文档访问范围")


def test_employee_default_metadata_filter() -> None:
    """
    测试普通员工不传 metadata_filter 时，系统自动加公共文档过滤条件。
    """

    metadata_filter = build_effective_metadata_filter(
        user_role="employee",
        request_metadata_filter=None,
    )

    assert metadata_filter is not None
    assert "source" in metadata_filter
    assert "$in" in metadata_filter["source"]

    allowed_sources = metadata_filter["source"]["$in"]

    assert "employee_handbook.md" in allowed_sources
    assert "travel_policy.md" in allowed_sources
    assert "hr_salary_policy.md" not in allowed_sources

    print_pass("employee 不传 metadata_filter 时自动限制为公共文档")


def test_hr_can_access_hr_private_doc() -> None:
    """
    测试 HR 可以访问 HR 私有文档。
    """

    metadata_filter = build_effective_metadata_filter(
        user_role="hr",
        request_metadata_filter={
            "source": "hr_salary_policy.md",
        },
    )

    assert metadata_filter == {
        "source": "hr_salary_policy.md",
    }

    print_pass("hr 可以访问 HR 私有文档")


def test_employee_cannot_access_hr_private_doc() -> None:
    """
    测试普通员工不能访问 HR 私有文档。
    """

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="employee",
            request_metadata_filter={
                "source": "hr_salary_policy.md",
            },
        ),
        "employee 不能访问 HR 私有文档",
    )


def test_cross_department_access_denied() -> None:
    """
    测试部门之间不能互相访问私有文档。
    """

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="hr",
            request_metadata_filter={
                "source": "finance_budget_policy.md",
            },
        ),
        "hr 不能访问财务私有文档",
    )

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="finance",
            request_metadata_filter={
                "source": "it_admin_policy.md",
            },
        ),
        "finance 不能访问 IT 私有文档",
    )

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="it",
            request_metadata_filter={
                "source": "hr_salary_policy.md",
            },
        ),
        "it 不能访问 HR 私有文档",
    )


def test_source_in_filter_is_intersected() -> None:
    """
    测试 source 使用 $in 时，系统会自动取用户有权限的交集。

    例如 HR 请求：
    [
        "employee_handbook.md",
        "hr_salary_policy.md",
        "finance_budget_policy.md"
    ]

    最终应该只保留：
    [
        "employee_handbook.md",
        "hr_salary_policy.md"
    ]
    """

    metadata_filter = build_effective_metadata_filter(
        user_role="hr",
        request_metadata_filter={
            "source": {
                "$in": [
                    "employee_handbook.md",
                    "hr_salary_policy.md",
                    "finance_budget_policy.md",
                ]
            }
        },
    )

    assert metadata_filter is not None
    assert metadata_filter["source"]["$in"] == [
        "employee_handbook.md",
        "hr_salary_policy.md",
    ]

    print_pass("source.$in 会自动过滤掉无权限文档")


def test_source_in_filter_all_denied() -> None:
    """
    测试 source.$in 中全部都是无权限文档时，直接拒绝。
    """

    assert_permission_denied(
        lambda: build_effective_metadata_filter(
            user_role="employee",
            request_metadata_filter={
                "source": {
                    "$in": [
                        "hr_salary_policy.md",
                        "finance_budget_policy.md",
                    ]
                }
            },
        ),
        "source.$in 全部无权限时返回 PERMISSION_DENIED",
    )


def test_admin_keep_request_filter() -> None:
    """
    测试 admin 会保留用户传入的 metadata_filter，不额外限制。
    """

    request_filter = {
        "source": "hr_salary_policy.md",
    }

    metadata_filter = build_effective_metadata_filter(
        user_role="admin",
        request_metadata_filter=request_filter,
    )

    assert metadata_filter == request_filter

    print_pass("admin 保留原始 metadata_filter")


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
    test_hr_can_access_hr_private_doc()
    test_employee_cannot_access_hr_private_doc()
    test_cross_department_access_denied()

    test_source_in_filter_is_intersected()
    test_source_in_filter_all_denied()

    test_admin_keep_request_filter()
    test_unknown_role_denied()

    print("=" * 100)
    print("权限控制单元测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)