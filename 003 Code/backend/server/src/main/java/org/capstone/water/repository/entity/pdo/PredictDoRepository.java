package org.capstone.water.repository.entity.pdo;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface PredictDoRepository extends JpaRepository<PredictDo, String> {
    //List<PdoMapping> findPredictDosByTankid(String tankid);
    boolean existsByTime(LocalDateTime time);

    @Query(value = "SELECT * FROM nursery.predictdo WHERE MOD(DATE_FORMAT(time,'%i'),30)=0 AND tankid = :id ORDER BY time desc limit 24", nativeQuery =true )
    List<PdoMapping> findPredictDosByTankid(@Param("id") String tankid);
}
