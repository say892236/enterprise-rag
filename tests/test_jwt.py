from api.auth.jwt import (
    create_access_token,
    decode_access_token,
)



def main():

    # 模拟登录成功后的用户信息
    user_data = {

        "sub": "ay",

        "user_id": 10001,

        "role": "employee",

    }


    # 创建token

    token = create_access_token(
        user_data
    )


    print("=" * 50)

    print("生成 Token:")

    print(token)



    print("=" * 50)


    # 解析token

    payload = decode_access_token(
        token
    )


    print("解析结果:")

    print(payload)



if __name__ == "__main__":

    main()